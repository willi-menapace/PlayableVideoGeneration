from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn
from dataset.video_dataset import VideoDataset
from evaluation.metrics.action_linear_classification import ActionClassificationScore
from evaluation.metrics.action_variance import ActionVariance
from evaluation.metrics.detection_metric_2d import DetectionMetric2D
from evaluation.metrics.fid import FID
from evaluation.metrics.fvd import FVD, IncrementalFVD
from evaluation.metrics.lpips import LPIPS
from evaluation.metrics.motion_masked_mse import MotionMaskedMSE
from evaluation.metrics.mse import MSE
from evaluation.metrics.psnr import PSNR
from evaluation.metrics.ssim import SSIM
from evaluation.metrics.tennis_player_detector import TennisPlayerDetector
from evaluation.metrics.vgg_cosine_similarity import VGGCosineSimilarity
from evaluation.plotting.density_plot import DensityPlotter
from evaluation.plotting.density_plot_2d import DensityPlotter2D
from evaluation.plotting.density_plot_2d_merged import DensityPlotter2DMerged
from evaluation.plotting.mean_vector_plot_2d import MeanVectorPlotter2D
from utils.logger import Logger
from utils.metrics_accumulator import MetricsAccumulator


class DatasetEvaluator:
    '''
    Generation results evaluator class
    '''

    def __init__(self, config, logger: Logger, reference_dataset: VideoDataset, generated_dataset: VideoDataset):
        '''
        Creates an evaluator

        :param config: The configuration file
        :param reference_dataset: the dataset to use as ground truth
        :param generated_dataset: the generated dataset to compare to ground truth
        '''

        self.config = config
        self.logger = logger
        self.reference_dataset = reference_dataset
        self.generated_dataset = generated_dataset
        self.reference_dataloader = DataLoader(self.reference_dataset,
                                               batch_size=self.config["evaluation"]["batching"]["batch_size"],
                                               shuffle=False, collate_fn=single_batch_elements_collate_fn,
                                               num_workers=self.config["evaluation"]["batching"]["num_workers"],
                                               pin_memory=True)
        self.generated_dataloader = DataLoader(self.generated_dataset,
                                               batch_size=self.config["evaluation"]["batching"]["batch_size"],
                                               shuffle=False, collate_fn=single_batch_elements_collate_fn,
                                               num_workers=self.config["evaluation"]["batching"]["num_workers"],
                                               pin_memory=True)

        if len(self.reference_dataloader) != len(self.generated_dataloader):
            raise Exception(f"Reference and generated datasets should have the same sequences, but their length differs:"
                            f"Reference ({len(self.reference_dataloader)}), Generated({len(self.generated_dataloader)})")

        self.detection_metric_2d = DetectionMetric2D()
        self.player_detector = TennisPlayerDetector()
        self.action_variance = ActionVariance()
        self.action_accuracy = ActionClassificationScore()

        self.mse = MSE()
        self.motion_masked_mse = MotionMaskedMSE()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.vgg_sim = VGGCosineSimilarity()
        self.fid = FID()
        #self.inception_score = InceptionScore()
        self.fvd = IncrementalFVD()

    def check_range(self, images):
        '''
        Checks that the images have the expected range
        :param images: (...) tensor with images
        :return:
        '''

        max = torch.max(images).item()
        min = torch.min(images).item()
        if max > 1.0 or min < 0.0:
            raise Exception(f"Input tensor outside allowed range [0.0, 1.0]: [{min}, {max}]")

    def compute_positional_statistics(self, values: np.ndarray, prefix:str) -> Dict:
        '''
        Computes statistics per each position in the sequence plus the statistic averaged over all positions

        :param values: (bs, sequence_length) tensor with input values
        :param prefix: String to use as prefix for each returned key
        :return: Dictionary with keys in the form (prefix/i: value) where i is the positional index or avg for the average
        '''

        results = {}

        # Computes the statistics per postion
        positional_values = values.mean(axis=0)
        positional_variances = values.var(axis=0).tolist()
        global_variance = float(positional_values.var())
        positional_values = positional_values.tolist()
        global_value = float(sum(positional_values) / len(positional_values))

        results[f"{prefix}/avg"] = global_value
        results[f"{prefix}/var"] = global_variance
        for idx, current_value in enumerate(positional_values):
            results[f"{prefix}/{idx}"] = current_value
        for idx, current_variance in enumerate(positional_variances):
            results[f"{prefix}/{idx}/var"] = current_variance

        return results

    def compute_movements_and_actions(self, reference_detections: np.ndarray, generated_batch):
        '''


        :param reference_detections: (bs, observations_count, 2) tensor with x and y coordinates of the detection, -1 if any
        :param generated_batch: batch of generated data
        :return: (detected_movements, 2), (detected_movements) arrays with detected movements,
                 actions inferred by the model corresponding to the detected movements
        '''
        movements = []
        inferred_actions = []

        batch_size = reference_detections.shape[0]
        observations_count = reference_detections.shape[1]

        for sequence_idx in range(batch_size):
            for observation_idx in range(observations_count - 1):
                # If there was a successful detection for both the current and the successive frame
                if reference_detections[sequence_idx, observation_idx, 0] != -1 and \
                   reference_detections[sequence_idx, observation_idx + 1, 0] != -1:
                    # Extract movement and action
                    current_movement = reference_detections[sequence_idx, observation_idx + 1] - reference_detections[sequence_idx, observation_idx]
                    current_action = generated_batch.video[sequence_idx].metadata[:-1][observation_idx]["inferred_action"]

                    movements.append(current_movement)
                    inferred_actions.append(current_action)

        return np.asarray(movements, dtype=np.float), np.asarray(inferred_actions, dtype=np.int)


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
                if idx % 1 == 0:
                    self.logger.print(f"- Computing metrics for batch [{idx}/{batches}]")

                # Extracts reference data
                reference_batch_tuple = reference_batch.to_tuple()
                reference_observations, reference_actions, reference_rewards, reference_dones = reference_batch_tuple

                # Extracts generated data
                generated_batch_tuple = generated_batch.to_tuple()
                generated_observations, generated_actions, generated_rewards, generated_dones = generated_batch_tuple

                reference_detections = self.player_detector(reference_observations)
                generated_detections = self.player_detector(generated_observations)

                # Checks the range of the input tensors
                self.check_range(reference_observations)
                self.check_range(generated_observations)

                # Computes metrics
                mse = self.mse(reference_observations, generated_observations)
                motion_masked_mse = self.motion_masked_mse(reference_observations, generated_observations)
                psnr = self.psnr(reference_observations, generated_observations)
                ssim = self.ssim(reference_observations, generated_observations)
                lpips = self.lpips(reference_observations, generated_observations)
                vgg_sim = self.vgg_sim(reference_observations, generated_observations)

                movements, inferred_actions = self.compute_movements_and_actions(reference_detections, generated_batch)

                accumulator.add("reference_detections", reference_detections)
                accumulator.add("generated_detections", generated_detections)
                accumulator.add("mse", mse.cpu().numpy())
                accumulator.add("motion_masked_mse", motion_masked_mse.cpu().numpy())
                accumulator.add("psnr", psnr.cpu().numpy())
                accumulator.add("ssim", ssim.cpu().numpy())
                accumulator.add("lpips", lpips.cpu().numpy())
                accumulator.add("vgg_sim", vgg_sim.cpu().numpy())
                accumulator.add("inferred_actions", inferred_actions)
                accumulator.add("movements", movements)

        # Obtains the computed values for each observation in the dataset
        mse_values = accumulator.pop("mse")
        motion_masked_mse_values = accumulator.pop("motion_masked_mse")
        psnr_values = accumulator.pop("psnr")
        ssim_values = accumulator.pop("ssim")
        lpips_values = accumulator.pop("lpips")
        vgg_sim_values = accumulator.pop("vgg_sim")
        inferred_actions = accumulator.pop("inferred_actions")
        movements = accumulator.pop("movements")

        all_reference_detections = accumulator.pop("reference_detections")
        all_generated_detections = accumulator.pop("generated_detections")

        self.logger.print("- Computing detection score")
        detection_results = self.detection_metric_2d(all_reference_detections, all_generated_detections, "detection")

        # Computes action variance and plots it
        plots_directory = self.config["logging"]["evaluation_images_directory"]
        actions_count = self.config["data"]["actions_count"]
        action_variance_results = self.action_variance(inferred_actions, movements, actions_count)
        action_accuracy_results = self.action_accuracy(inferred_actions, movements, actions_count)
        DensityPlotter.plot_density(inferred_actions, movements, actions_count, plots_directory)
        DensityPlotter2D.plot_density(inferred_actions, movements, actions_count, plots_directory)
        DensityPlotter2DMerged.plot_density(inferred_actions, movements, actions_count, plots_directory)
        MeanVectorPlotter2D.plot(inferred_actions, movements, actions_count, plots_directory)

        self.logger.print("- Computing FID score")
        fid_result = self.fid(self.reference_dataloader, self.generated_dataloader)

        self.logger.print("- Computing FVD score")
        fvd_result = self.fvd(self.reference_dataloader, self.generated_dataloader)

        results = {}
        # Computes the results for each position in the sequence
        mse_results = self.compute_positional_statistics(mse_values, "mse")
        motion_masked_mse_results = self.compute_positional_statistics(motion_masked_mse_values, "motion_masked_mse")
        psnr_results = self.compute_positional_statistics(psnr_values, "psnr")
        ssim_results = self.compute_positional_statistics(ssim_values, "ssim")
        lpips_results = self.compute_positional_statistics(lpips_values, "lpips")
        vgg_sim_results = self.compute_positional_statistics(vgg_sim_values, "vgg_sim")

        # Merges all the results
        results = dict(results, **mse_results)
        results = dict(results, **motion_masked_mse_results)
        results = dict(results, **psnr_results)
        results = dict(results, **ssim_results)
        results = dict(results, **lpips_results)
        results = dict(results, **vgg_sim_results)
        results = dict(results, **detection_results)
        results = dict(results, **action_variance_results)
        results = dict(results, **action_accuracy_results)
        results["fid"] = fid_result
        results["fvd"] = fvd_result

        return results


def evaluator(config, logger: Logger, reference_dataset: VideoDataset, generated_dataset: VideoDataset):
    return DatasetEvaluator(config, logger, reference_dataset, generated_dataset)