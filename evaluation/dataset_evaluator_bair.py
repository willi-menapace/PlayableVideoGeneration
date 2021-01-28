from typing import Dict

import torch
import numpy as np

from dataset.video_dataset import VideoDataset
from evaluation.dataset_evaluator import DatasetEvaluator
from evaluation.plotting.density_plot import DensityPlotter
from evaluation.plotting.density_plot_2d import DensityPlotter2D
from evaluation.plotting.density_plot_2d_merged import DensityPlotter2DMerged
from evaluation.plotting.mean_vector_plot_2d import MeanVectorPlotter2D
from utils.metrics_accumulator import MetricsAccumulator


class DatasetEvaluatorBair(DatasetEvaluator):
    '''
    Generation results evaluator class
    '''

    def __init__(self, config, logger, reference_dataset: VideoDataset, generated_dataset: VideoDataset):
        '''
        Creates an evaluator

        :param config: The configuration file
        :param logger: The object to use for logging
        :param reference_dataset: the dataset to use as ground truth
        :param generated_dataset: the generated dataset to compare to ground truth
        '''

        super(DatasetEvaluatorBair, self).__init__(config, logger, reference_dataset, generated_dataset)

    def compute_metrics(self) -> Dict:
        '''
        Computes evaluation metrics on the given datasets

        :return: Dictionary with evaluation results
        '''

        accumulator = MetricsAccumulator()

        batches = len(self.reference_dataloader)
        with torch.no_grad():
            for idx, (reference_batch, generated_batch) in enumerate(zip(self.reference_dataloader, self.generated_dataloader)):

                # Logs the current batch
                if idx % 10 == 0:
                    self.logger.print(f"- Computing metrics for batch [{idx}/{batches}]")

                # Extracts reference data
                reference_batch_tuple = reference_batch.to_tuple()
                reference_observations, reference_actions, reference_rewards, reference_dones = reference_batch_tuple

                # Extracts generated data
                generated_batch_tuple = generated_batch.to_tuple()
                generated_observations, generated_actions, generated_rewards, generated_dones = generated_batch_tuple

                batch_size = generated_observations.size(0)
                observations_count = generated_observations.size(1)
                inferred_actions = []
                movements = []
                for sequence_idx in range(batch_size):
                    # Extracts the inferred actions in the current generated sequence
                    current_inferred_actions = [current_metadata["inferred_action"] for current_metadata in generated_batch.video[sequence_idx].metadata[:-1]]
                    current_inferred_actions = np.asarray(current_inferred_actions)
                    inferred_actions.append(current_inferred_actions)

                    # Extracts the hand movements in the reference sequence
                    current_movements = []
                    for element_idx in range(1, observations_count):
                        predecessor_state = np.asarray(reference_batch.video[sequence_idx].metadata[element_idx - 1]["state"])
                        successor_state = np.asarray(reference_batch.video[sequence_idx].metadata[element_idx]["state"])
                        current_movements.append(successor_state - predecessor_state)
                    movements.append(current_movements)

                # Converts to numpy
                inferred_actions = np.asarray(inferred_actions)
                movements = np.asarray(movements)

                # Checks the range of the input tensors
                self.check_range(reference_observations)
                self.check_range(generated_observations)

                # Computes metrics
                mse = self.mse(reference_observations, generated_observations)
                psnr = self.psnr(reference_observations, generated_observations)
                ssim = self.ssim(reference_observations, generated_observations)
                lpips = self.lpips(reference_observations, generated_observations)
                vgg_sim = self.vgg_sim(reference_observations, generated_observations)

                accumulator.add("mse", mse.cpu().numpy())
                accumulator.add("psnr", psnr.cpu().numpy())
                accumulator.add("ssim", ssim.cpu().numpy())
                accumulator.add("lpips", lpips.cpu().numpy())
                accumulator.add("vgg_sim", vgg_sim.cpu().numpy())
                accumulator.add("inferred_actions", inferred_actions)
                accumulator.add("movements", movements)

        # Obtains the computed values for each observation in the dataset
        mse_values = accumulator.pop("mse")
        psnr_values = accumulator.pop("psnr")
        ssim_values = accumulator.pop("ssim")
        lpips_values = accumulator.pop("lpips")
        vgg_sim_values = accumulator.pop("vgg_sim")
        inferred_actions = accumulator.pop("inferred_actions")
        movements = accumulator.pop("movements")

        results = {}
        # Computes the results for each position in the sequence
        mse_results = self.compute_positional_statistics(mse_values, "mse")
        psnr_results = self.compute_positional_statistics(psnr_values, "psnr")
        ssim_results = self.compute_positional_statistics(ssim_values, "ssim")
        lpips_results = self.compute_positional_statistics(lpips_values, "lpips")
        vgg_sim_results = self.compute_positional_statistics(vgg_sim_values, "vgg_sim")

        # Computes action variance and plots it
        plots_directory = self.config["logging"]["evaluation_images_directory"]
        actions_count = self.config["data"]["actions_count"]
        action_variance_results = self.action_variance(inferred_actions, movements, actions_count)
        try:
            action_accuracy_results = self.action_accuracy(inferred_actions, movements, actions_count)
        except Exception as e:
            print("Warning: action accuracy results could not be computed due to an unexpected error")
            print(e)
            action_accuracy_results = {}
        DensityPlotter.plot_density(inferred_actions, movements, actions_count, plots_directory)
        DensityPlotter2D.plot_density(inferred_actions, movements, actions_count, plots_directory, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), axis_inversion=True)
        DensityPlotter2DMerged.plot_density(inferred_actions, movements, actions_count, plots_directory, xlim=(-0.1, 0.1), ylim=(-0.1, 0.1), axis_inversion=True)
        MeanVectorPlotter2D.plot(inferred_actions, movements, actions_count, plots_directory, xlim=(-0.05, 0.05), ylim=(-0.05, 0.05), axis_inversion=True)

        self.logger.print("- Computing FID score")
        fid_result = self.fid(self.reference_dataloader, self.generated_dataloader)

        del mse_values
        del psnr_values
        del ssim_values
        del vgg_sim_values
        del inferred_actions
        del movements
        self.logger.print("- Computing FVD score")
        fvd_result = self.fvd(self.reference_dataloader, self.generated_dataloader)

        # Merges all the results
        results = dict(results, **mse_results)
        results = dict(results, **psnr_results)
        results = dict(results, **ssim_results)
        results = dict(results, **lpips_results)
        results = dict(results, **vgg_sim_results)
        results = dict(results, **action_variance_results)
        results = dict(results, **action_accuracy_results)
        results["fid"] = fid_result
        results["fvd"] = fvd_result

        return results


def evaluator(config, logger, reference_dataset: VideoDataset, generated_dataset: VideoDataset):
    return DatasetEvaluatorBair(config, logger, reference_dataset, generated_dataset)