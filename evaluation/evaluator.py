import os
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
import matplotlib.cm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from PIL import Image

from sklearn.utils.linear_assignment_ import linear_assignment

from dataset.batching import single_batch_elements_collate_fn
from dataset.video_dataset import VideoDataset
from training.losses import KLGaussianDivergenceLoss, MutualInformationLoss, ParallelPerceptualLoss, \
    SequenceLossEvaluator, MotionLossWeightMaskCalculator, EntropyProbabilityLoss, ObservationsLoss, PerceptualLoss, \
    StatesLoss, EntropyLogitLoss
from utils.average_meter import AverageMeter
from utils.tensor_displayer import TensorDisplayer
from utils.tensor_folder import TensorFolder
from utils.tensor_resizer import TensorResizer


class Evaluator:
    '''
    Helper class for model evaluation
    '''

    def __init__(self, config, dataset: VideoDataset, logger, action_sampler, logger_prefix="test"):
        '''

        :param config: main configuration file
        :param dataset: video dataset to use for evaluation
        :param logger: logger to use for evaluation
        :param action_sampler: ActionSampler object to use for selecting actions during evaluation
        :param logger_prefix: prefix to use for all the logged values
        '''

        self.config = config
        self.logger = logger
        self.logger_prefix = logger_prefix
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.config["evaluation"]["batching"]["batch_size"], shuffle=False, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)
        self.imaging_dataloader = DataLoader(dataset, batch_size=self.config["evaluation"]["batching"]["batch_size"], shuffle=True, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)

        # Defines the losses
        self.observations_loss = ObservationsLoss()
        self.states_loss = StatesLoss()
        self.entropy_loss = EntropyLogitLoss()
        self.samples_entropy_loss = EntropyProbabilityLoss()
        self.observations_perceptual_loss = ParallelPerceptualLoss()
        self.action_directions_kl_gaussian_divergence_loss = KLGaussianDivergenceLoss()
        self.mutual_information_loss = MutualInformationLoss()
        self.action_distribution_entropy = EntropyProbabilityLoss()

        # Defines the losses to apply to the sequences
        self.sequence_observation_loss = SequenceLossEvaluator(self.observations_loss)
        self.sequence_perceptual_loss = SequenceLossEvaluator(self.observations_perceptual_loss)
        self.sequence_states_loss = SequenceLossEvaluator(self.states_loss)

        self.max_evaluation_batches = self.config["evaluation"]["max_evaluation_batches"]

        # Defines the calculator for motion weights
        self.weight_mask_calculator = MotionLossWeightMaskCalculator(self.config["training"]["motion_weights_bias"])

        self.action_sampler = action_sampler
        self.best_action_mappings = None

    def set_action_sampler(self, action_sampler):
        '''
        Sets the action sampler to use during evaluation

        :param action_sampler: ActionSampler object to use for selecting actions during evaluation
        :return:
        '''

        self.action_sampler = action_sampler

    def get_best_action_mappings(self) -> Dict[int, int]:
        '''
        Returns the latest map from ground truth actions to corresponding indexes in the model action space.
        Can only be called after an invocation of evaluate
        :return:
        '''

        if self.best_action_mappings is None:
            raise Exception("The action mapping can be computed only after a call to evaluate")

        return self.best_action_mappings

    def evaluate(self, model, step: int):
        '''
        Evaluates the performances of the given model

        :param model: The model to evaluate
        :param step: The current step
        :return:
        '''

        loss_averager = AverageMeter()

        # All the selected actions and ground truth ones
        all_gt_actions = []
        all_pred_actions = []

        # Number of video sequence samples analyzed
        total_sequences = 0

        self.logger.print(f"== Evaluation [{step}][{self.logger_prefix}] ==")

        self.logger.print(f"- Saving sample images")
        # Saves sample images
        with torch.no_grad():
            for idx, batch in enumerate(self.imaging_dataloader):

                # Performs inference
                batch_tuple = batch.to_tuple()
                observations, actions, rewards, dones = batch_tuple

                ground_truth_observations = batch_tuple[0]
                results = model(batch_tuple, ground_truth_observations_init=1)
                reconstructed_observations, multiresolution_reconstructed_observations, reconstructed_states, states, hidden_states, selected_actions, action_logits, action_samples_distribution, *others = results

                weights_mask = self.weight_mask_calculator.compute_weight_mask(ground_truth_observations,
                                                                                                reconstructed_observations)
                # Saves reconstructed observations at all resolutions
                for resolution_idx, current_reconstructed_observations in enumerate(multiresolution_reconstructed_observations):
                    self.save_examples(observations, current_reconstructed_observations, step, max_batches=30, log_key=f"observations_r{resolution_idx}")

                # Plots weight masks
                assert (weights_mask.min().item() >= 0.0)
                weights_mask = weights_mask / torch.max(torch.abs(weights_mask))  # Normalizes the weights for plotting
                self.save_examples_with_weights(observations, weights_mask, reconstructed_observations, weights_mask,
                                                step, max_batches=30, log_key="motion_weighted_observations_")

                # If attention is used, plot also attention
                if len(others) > 0:
                    attention = others[0]
                    reconstructed_attention = others[1]
                    self.save_examples_with_weights(observations, attention, reconstructed_observations,
                                                    reconstructed_attention, step, max_batches=30, log_key="attentive_observations_")

                break

        self.logger.print(f"- Computing evaluation losses")
        current_evaluation_batches = 0
        all_action_direction_distributions = []
        all_action_logits = []
        all_action_states = []
        estimated_action_centroids = None
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                if self.max_evaluation_batches is not None and self.max_evaluation_batches <= current_evaluation_batches:
                    self.logger.print(f"- Aborting evaluation, maximum number of evaluation batches reached")
                    break
                current_evaluation_batches += 1

                # Performs evaluation only on the plain batch
                total_sequences += batch.size

                # Performs inference
                batch_tuple = batch.to_tuple()
                results = model(batch_tuple, ground_truth_observations_init=1, action_sampler=self.action_sampler)

                # Extracts the results
                reconstructed_observations, multiresolution_reconstructed_observations, reconstructed_states, states, hidden_states, selected_actions, action_logits, action_samples_distribution, \
                attention, reconstructed_attention, action_directions_distribution, sampled_action_directions, \
                action_states_distribution, sampled_action_states, action_variations,\
                reconstructed_action_logits, \
                reconstructed_action_directions_distribution, reconstructed_sampled_action_directions, \
                reconstructed_action_states_distribution, reconstructed_sampled_action_states, \
                *other_results = results

                all_action_states.append(action_states_distribution[:, :, :, 0])
                all_action_direction_distributions.append(action_directions_distribution.cpu())
                all_action_logits.append(action_logits.cpu())
                if estimated_action_centroids is None:
                    estimated_action_centroids = model.module.centroid_estimator.get_estimated_centroids()

                # Computes losses
                entropy_loss = self.entropy_loss(action_logits)
                samples_entropy = self.samples_entropy_loss(action_samples_distribution)
                action_ditribution_entropy = self.action_distribution_entropy(action_samples_distribution.mean(dim=(0, 1)).unsqueeze(dim=0))
                action_directions_kl_divergence_loss = self.action_directions_kl_gaussian_divergence_loss(action_directions_distribution)
                action_mutual_information_loss = self.mutual_information_loss(torch.softmax(action_logits, dim=-1), torch.softmax(reconstructed_action_logits, dim=-1))

                # Evaluates the sequence losses
                sequence_observation_loss = self.evaluate_loss_on_sequence(batch.observations, reconstructed_observations, self.sequence_observation_loss, "observations_loss")
                sequence_perceptual_loss = self.evaluate_loss_on_sequence(batch.observations, reconstructed_observations, self.sequence_perceptual_loss, "perceptual_loss")
                sequence_states_loss = self.evaluate_loss_on_sequence(states, reconstructed_states, self.sequence_states_loss, "states_loss")

                loss_averager.add(sequence_observation_loss)
                loss_averager.add(sequence_perceptual_loss)
                loss_averager.add(sequence_states_loss)
                loss_averager.add({"entropy": entropy_loss.item()})
                loss_averager.add({"samples_entropy": samples_entropy.item()})
                loss_averager.add({"action_distribution_entropy": action_ditribution_entropy.item()})
                loss_averager.add({"action_directions_kl_loss": action_directions_kl_divergence_loss.item()})
                loss_averager.add({"action_mutual_information_loss": action_mutual_information_loss.item()})

                # Saves the flattened actions
                all_pred_actions.append(selected_actions.reshape((-1)))
                all_gt_actions.append(batch.actions[:, :-1].reshape((-1))) # The last action of each sequence cannot be predicted

        all_action_states = torch.cat(all_action_states)
        all_predecessor_action_states = TensorFolder.flatten(all_action_states[:, :-1])
        all_successor_action_states = TensorFolder.flatten(all_action_states[:, 1:])
        samples = torch.cat([all_predecessor_action_states, all_successor_action_states], dim=-1).cpu().numpy()
        covariance_matrix = np.cov(samples, rowvar=False)

        all_pred_actions = torch.cat(all_pred_actions) # Concatenate on the batch size dimension
        all_gt_actions = torch.cat(all_gt_actions)

        actions_accuracy, actions_match = self.compute_actions_accuracy(all_pred_actions, all_gt_actions)

        # Plots the distribution of action directions
        all_action_direction_distributions = torch.cat(all_action_direction_distributions, dim=0)
        all_action_logits = torch.cat(all_action_logits, dim=0)
        all_action_probabilities = torch.softmax(all_action_logits, dim=-1)
        action_directions_plot_filename = os.path.join(self.config["logging"]["output_images_directory"], f"action_direction_space_eval_{step}.pdf")
        TensorDisplayer.show_action_directions(estimated_action_centroids.detach().cpu(), all_action_direction_distributions, all_action_probabilities, action_directions_plot_filename)

        # Registers the best match found
        self.best_action_mappings = actions_match

        # Populates data to log at the current step
        log_data = {f"{self.logger_prefix}/actions_accuracy": actions_accuracy,
                    "step": step}
        for key in loss_averager.data:
            log_data[f'{self.logger_prefix}/{key}'] = loss_averager.pop(key)

        # Logs results
        wandb = self.logger.get_wandb()
        wandb.log(log_data, step=step)

        self.logger.print("- observations_loss: {:.3f}".format(log_data[f'{self.logger_prefix}/observations_loss/avg']))
        self.logger.print("- perceptual_loss: {:.3f}".format(log_data[f'{self.logger_prefix}/perceptual_loss/avg']))
        self.logger.print("- states_loss: {:.3f}".format(log_data[f'{self.logger_prefix}/states_loss/avg']))
        self.logger.print("- actions_accuracy: {:.3f}".format(actions_accuracy))
        self.logger.print("- entropy: {:.3f}".format(log_data[f'{self.logger_prefix}/entropy']))
        self.logger.print("- samples entropy: {:.3f}".format(log_data[f'{self.logger_prefix}/samples_entropy']))
        self.logger.print("- action distribution entropy: {:.3f}".format(log_data[f'{self.logger_prefix}/action_distribution_entropy']))

        return

    def evaluate_loss_on_sequence(self, ground_truth_sequence: torch.Tensor, reconstructed_sequence: torch.Tensor, loss: SequenceLossEvaluator, prefix: str) -> Dict:
        '''
        Evaluates a loss at each position in the given sequence

        :param ground_truth_sequence: (bs, sequence_length, ...) tensor with ground truth values
        :param reconstructed_sequence: (bs, sequence_length, ...) tensor with reconstructed values
        :param loss: the loss function to call
        :return: map in the form key -> value with evaluation results over the whole sequence
        '''

        results = {}
        avg_loss, all_losses = loss(ground_truth_sequence, reconstructed_sequence)

        # Unpacks the average loss and the loss computed for each sequence position
        results[f'{prefix}/avg'] = avg_loss.item()
        for idx, current_loss in enumerate(all_losses):
            results[f'{prefix}/pos_{idx}'] = current_loss.item()

        return results

    def upscale_and_color_weights(self, weights, height, width):
        '''
        :param weights: (bs, observations_count, 1, states_height, states_width) tensor with weights over observed images
        :param height: the upscaled weights height
        :param width: the upscaled weights width
        :return: (bs, observations_count, 3, height, width) tensor with colormapped weights
        '''

        batch_size = weights.size(0)
        observations_count = weights.size(1)
        weights_height = weights.size(3)
        weights_width = weights.size(4)
        colormap = matplotlib.cm.get_cmap('viridis')

        weights = weights.cpu().numpy()

        all_colored_images = []
        for current_batch_index in range(batch_size):
            for current_observation_index in range(observations_count):
                # Colors the frame and converts it to torch format
                current_frame = weights[current_batch_index, current_observation_index, 0]
                current_colored_frame = colormap(current_frame)[:,:,0:-1]
                all_colored_images.append(torch.from_numpy(current_colored_frame).permute([2, 0, 1]).cuda())

        # Rescreates the original tensor which now has 3 channels due to color
        weights = torch.stack(all_colored_images, dim=0)
        # Upscales the weights if needed
        if height != weights_height or width != weights_width:
            weights = F.interpolate(weights, size=(height, width), mode='bilinear', align_corners=False)
        weights = weights.reshape(batch_size, observations_count, 3, height, width)

        return weights

    def blend_tensors(self, first: torch.Tensor, second: torch.Tensor, blend_factor: float):
        """
        Blends the two tensor with a certain transparency factor

        :param first: the first tensor to blend
        :param second: the second tensor to blend with same dimension as the first
        :param blend_factor: percentage of the second frame that is blended into the first
        :return: tensor representing the blended frame of dimension equal to the original tensors
        """
        return first * (1 - blend_factor) + second * blend_factor

    def save_examples_with_weights(self, observations: torch.Tensor, weights: torch.Tensor,
                                   reconstructed_observations: torch.Tensor, reconstructed_weights: torch.Tensor,
                                   step, blend_factor=0.6, log_key="observations", max_batches=100):
        '''
        Saves images showing observations and corresponding reconstructed observations

        :param observations: (bs, observations_count, 3 * observations_stacking, h, w) tensor with observed images
        :param weights: (bs, observations_count, 1, weights_height, weights_width) tensor with weights over observed images
        :param reconstructed_observations: (bs, observations_count, 3, h, w) tensor with reconstructed frames
        :param reconstructed_weights: (bs, observations_count|observations_count-1, 1, states_height, states_width) tensor with weights over reconstructed images
        :param log_key: key to use for logging
        :return:
        '''

        observations = observations[:, :, :3]  # For each observation extract only the current frame and not the past ones
        # Resize the observations if needed
        observations = TensorResizer.resize_as(observations, reconstructed_observations)

        # Normalizes the range of the observations
        observations = self.check_and_normalize_range(observations)
        reconstructed_observations = self.check_and_normalize_range(reconstructed_observations)

        if observations.size(0) > max_batches:
            observations = observations[:max_batches]
            weights = weights[:max_batches]
            reconstructed_observations = reconstructed_observations[:max_batches]
            reconstructed_weights = reconstructed_weights[:max_batches]

        batch_size = observations.size(0)
        observations_count = observations.size(1)
        observations_height = observations.size(3)
        observations_width = observations.size(4)
        reconstructed_observations_count = reconstructed_observations.size(1)

        reconstructed_weights_sequence_length = reconstructed_weights.size(1)

        # If the first reconstructed observation is not available, use the ground truth one
        if reconstructed_observations_count == observations_count - 1:
            reconstructed_observations = torch.cat([observations[:, 0:1], reconstructed_observations], dim=1)
        if reconstructed_weights_sequence_length == observations_count - 1:
            reconstructed_weights = torch.cat([weights[:, 0:1], reconstructed_weights], dim=1)

        # Upscales and colorizes the weights
        colorized_weights = self.upscale_and_color_weights(weights, observations_height, observations_width)
        colorized_reconstructed_weights = self.upscale_and_color_weights(reconstructed_weights, observations_height, observations_width)

        colorized_observations = self.blend_tensors(observations, colorized_weights, blend_factor)
        colorized_reconstructed_observations = self.blend_tensors(reconstructed_observations, colorized_reconstructed_weights, blend_factor)

        observations_list = []
        # Disposes the images in alternating rows with originals on the top and reconstructed at the bottom
        for batch_element_idx in range(batch_size):
            for observation_element_idx in range(observations_count):
                observations_list.append(colorized_observations[batch_element_idx, observation_element_idx])
            for observation_element_idx in range(observations_count):
                observations_list.append(colorized_reconstructed_observations[batch_element_idx, observation_element_idx])

        observations_grid = np.transpose(make_grid(observations_list, padding=2, pad_value=1, nrow=observations_count).cpu().numpy(), (1, 2, 0))

        Image.fromarray((observations_grid * 255).astype(np.uint8)).save(os.path.join(self.config["logging"]["output_images_directory"], f"{step:09}_{log_key}.png"))

        wandb = self.logger.get_wandb()
        wandb.log({f"{self.logger_prefix}/{log_key}": wandb.Image(observations_grid), "step": step}, step=step)

    def check_and_normalize_range(self, observations: torch.Tensor) -> torch.Tensor:
        '''
        If the range of the observations is in [-1, 1] instead of [0, 1] it normalizes it
        :param observations: arbitrarily shaped tensor to normalize
        :return: the input tensor normalized in [0, 1]
        '''

        minimum_value = torch.min(observations).item()

        # Check range and normalize
        if minimum_value < 0:
            observations = (observations + 1) / 2
        return observations

    def save_examples(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, step, log_key="observations", max_batches=100):
        '''
        Saves images showing observations and corresponding reconstructed observations

        :param observations: (bs, observations_count, 3 * observations_stacking, h, w) tensor with observed images
        :param reconstructed_observations: (bs, observations_count, 3, h, w) tensor with reconstructed frames
        :param log_key: key to use for logging
        :return:
        '''

        observations = observations[:, :, :3]  # For each observation extract only the current frame and not the past ones

        # Resize the observations if needed
        observations = TensorResizer.resize_as(observations, reconstructed_observations)

        # Normalizes the range of the observations
        observations = self.check_and_normalize_range(observations)
        reconstructed_observations = self.check_and_normalize_range(reconstructed_observations)

        if observations.size(0) > max_batches:
            observations = observations[:max_batches]
            reconstructed_observations = reconstructed_observations[:max_batches]

        batch_size = observations.size(0)
        observations_count = observations.size(1)
        reconstructed_observations_count = reconstructed_observations.size(1)

        # If the first reconstructed observation is not available, use the ground truth one
        if reconstructed_observations_count == observations_count - 1:
            reconstructed_observations = torch.cat([observations[:, 0:1], reconstructed_observations], dim=1)

        observations_list = []
        # Disposes the images in alternating rows with originals on the top and reconstructed at the bottom
        for batch_element_idx in range(batch_size):
            for observation_element_idx in range(observations_count):
                observations_list.append(observations[batch_element_idx, observation_element_idx])
            for observation_element_idx in range(observations_count):
                observations_list.append(reconstructed_observations[batch_element_idx, observation_element_idx])

        observations_grid = np.transpose(make_grid(observations_list, padding=2, pad_value=1, nrow=observations_count).cpu().numpy(), (1, 2, 0))

        Image.fromarray((observations_grid * 255).astype(np.uint8)).save(os.path.join(self.config["logging"]["output_images_directory"], f"{step:09}_{log_key}.png"))

        wandb = self.logger.get_wandb()
        wandb.log({f"{self.logger_prefix}/{log_key}": wandb.Image(observations_grid), "step": step}, step=step)

    def compute_actions_accuracy(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Tuple:
        '''
        Computes the accuracy using the hungarian algorithm of the best match between predicted and ground truth actions
        :param predictions: (num_samples) tensor with predicted actions
        :param ground_truth: (num_samples) tensor with ground truth actions
        :return: (accuracy, actions_match) accuracy of the best possible mapping between predicted actions and ground truth actions
                                           and list of tuples (model_action_index, ground_truth_action_index)
        '''

        num_samples = predictions.shape[0]

        match = self._hungarian_match(predictions, ground_truth)

        found = torch.zeros(self.config["data"]["actions_count"])
        reordered_preds = torch.zeros(num_samples, dtype=predictions.dtype).cuda()

        for pred_i, target_i in match:
            # reordered_preds[flat_predss_all[i] == pred_i] = target_i
            reordered_preds[torch.eq(predictions, int(pred_i))] = torch.from_numpy(np.array(target_i)).cuda().int().item()
            found[pred_i] = 1
        assert (found.sum() == self.config["data"]["actions_count"])  # each output_k must get mapped

        ground_truth_to_model_mapping = {}
        for model_action_index, ground_truth_action_index in match:
            ground_truth_to_model_mapping[ground_truth_action_index] = int(model_action_index)

        return torch.sum(reordered_preds == ground_truth).item() / num_samples, ground_truth_to_model_mapping

    def _hungarian_match(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> List[Tuple[int, int]]:
        '''
        Performs the hungarian matching between prediction classes and ground truth classes
        :param predictions: (num_samples) tensor with predicted actions
        :param ground_truth: (num_samples) tensor with ground truth actions
        :return:
        '''
        assert (isinstance(predictions, torch.Tensor) and isinstance(ground_truth, torch.Tensor))

        num_samples = ground_truth.shape[0]

        num_k = self.config["data"]["actions_count"]
        num_correct = np.zeros((num_k, num_k))

        for c1 in range(num_k):
            for c2 in range(num_k):
                # elementwise, so each sample contributes once
                votes = int(((predictions == c1) * (ground_truth == c2)).sum())
                num_correct[c1, c2] = votes

        # num_correct is small
        match = linear_assignment(num_samples - num_correct)

        # return as list of tuples, out_c to gt_c
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))

        return res


def evaluator(config, dataset: VideoDataset, logger, action_sampler, logger_prefix="test"):
    return Evaluator(config, dataset, logger, action_sampler, logger_prefix)
