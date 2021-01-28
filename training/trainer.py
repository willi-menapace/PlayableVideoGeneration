import collections
import math
import os
from typing import Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image

from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn, Batch
from dataset.video_dataset import VideoDataset
from training.losses import ObservationsLoss, StatesLoss, EntropyLogitLoss, KLDivergence, HiddenStatesLoss, \
    EntropyProbabilityLoss, PerceptualLoss, MotionLossWeightMaskCalculator, KLGaussianDivergenceLoss, \
    MutualInformationLoss, ParallelPerceptualLoss, KLGeneralGaussianDivergenceLoss
from utils.average_meter import AverageMeter
from utils.logger import Logger
from utils.tensor_displayer import TensorDisplayer


class Trainer:
    '''
    Helper class for model training
    '''

    def __init__(self, config, model, dataset: VideoDataset, logger: Logger):

        self.config = config
        self.dataset = dataset
        self.logger = logger

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.config["training"]["lr_schedule"], gamma=self.config["training"]["lr_gamma"])

        self.dataloader = DataLoader(dataset, batch_size=self.config["training"]["batching"]["batch_size"], drop_last=True, shuffle=True, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["training"]["batching"]["num_workers"], pin_memory=True)

        # Initializes losses
        self.weight_mask_calculator = MotionLossWeightMaskCalculator(self.config["training"]["motion_weights_bias"])
        self.perceptual_loss = PerceptualLoss()
        self.observations_loss = ObservationsLoss()
        self.states_loss = StatesLoss()
        self.hidden_states_loss = HiddenStatesLoss()
        self.entropy_loss = EntropyLogitLoss()
        self.actions_divergence_loss = KLDivergence()
        self.samples_entropy_loss = EntropyProbabilityLoss()
        self.action_distribution_entropy = EntropyProbabilityLoss()
        self.perceptual_loss = ParallelPerceptualLoss()
        self.action_state_distribution_kl = KLGeneralGaussianDivergenceLoss()
        self.action_directions_kl_gaussian_divergence_loss = KLGaussianDivergenceLoss()
        self.mutual_information_loss = MutualInformationLoss()

        self.average_meter = AverageMeter()
        self.global_step = 0

        self.action_mutual_infromation_entropy_lambda = config["training"]["action_mutual_information_entropy_lambda"]

        # Observations count annealing parameters
        self.observations_count_start = self.config["training"]["batching"]["observations_count_start"]
        self.observations_count_end = self.config["training"]["batching"]["observations_count"]
        self.observations_count_steps = self.config["training"]["batching"]["observations_count_steps"]

        # Real observations annealing parameters
        self.real_observations_start = self.config["training"]["ground_truth_observations_start"]
        self.real_observations_end = self.config["training"]["ground_truth_observations_end"]
        self.real_observations_steps = self.config["training"]["ground_truth_observations_steps"]

        # Gumbel temperature annealing parameters
        self.gumbel_temperature_start = self.config["training"]["gumbel_temperature_start"]
        self.gumbel_temperature_end = self.config["training"]["gumbel_temperature_end"]
        self.gumbel_temperature_steps = self.config["training"]["gumbel_temperature_steps"]

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return (param_group['lr'])

    def save_checkpoint(self, model, name=None):
        '''
        Saves the current training state
        :param model: the model to save
        :param name: the name to give to the checkopoint. If None the default name is used
        :return:
        '''

        if name is None:
            filename = os.path.join(self.config["logging"]["save_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["save_root_directory"], f"{name}_.pth.tar")

        # If the model is wrapped, save the internal state
        is_data_parallel = isinstance(model, nn.DataParallel)
        if is_data_parallel:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save({"model": model_state_dict, "optimizer": self.optimizer.state_dict(), "lr_scheduler": self.lr_scheduler.state_dict(), "step": self.global_step}, filename)

    def load_checkpoint(self, model, name=None):
        """
        Loads the model from a saved state
        :param model: The model to load
        :param name: Name of the checkpoint to use. If None the default name is used
        :return:
        """

        if name is None:
            filename = os.path.join(self.config["logging"]["save_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["save_root_directory"], f"{name}.pth.tar")

        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        loaded_state = torch.load(filename)
        model.load_state_dict(loaded_state["model"])
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        self.global_step = loaded_state["step"]

    def get_ground_truth_observations_count(self) -> int:
        '''
        Computes the number of ground truth observations to use for the current training step according to the annealing
        parameters
        :return: number of ground truth observations to use in the training sequence at the current step
        '''

        ground_truth_observations_count = self.real_observations_start - \
                                  (self.real_observations_start - self.real_observations_end) * \
                                  self.global_step / self.real_observations_steps
        ground_truth_observations_count = math.ceil(ground_truth_observations_count)
        ground_truth_observations_count = max(self.real_observations_end, ground_truth_observations_count)

        return ground_truth_observations_count

    def get_gumbel_temperature(self) -> float:
        '''
        Computes the gumbel temperature to use at the current step
        :return: Gumbel temperature to use at the current step
        '''

        gumbel_temperature = self.gumbel_temperature_start - \
                                  (self.gumbel_temperature_start - self.gumbel_temperature_end) * \
                                  self.global_step / self.gumbel_temperature_steps
        gumbel_temperature = max(self.gumbel_temperature_end, gumbel_temperature)

        return gumbel_temperature

    def get_observations_count(self):
        '''
        Computes the number of observations to use for the sequence at the current training step according to
        the annealing parameters
        :return: Number of observations to use in each training sequence at the current step
        '''

        observations_count = self.observations_count_start + \
                                  (self.observations_count_end - self.observations_count_start) * \
                                  self.global_step / self.observations_count_steps
        observations_count = math.floor(observations_count)
        observations_count = min(self.observations_count_end, observations_count)

        return observations_count

    def sum_loss_components(self, components: List[torch.Tensor], weights: Union[List[float], float]) -> List[torch.Tensor]:
        '''
        Produces the weighted sum of the loss components

        :param components: List of scalar tensors
        :param weights: List of weights of the same length of components, or single weight to apply to each component
        :return: Weighted sum of the components
        '''

        components_count = len(components)

        # If the weight is a scalar, broadcast it
        if not isinstance(weights, collections.Sequence):
            weights = [weights] * components_count

        total_sum = components[0] * 0.0
        for current_component, current_weight in zip(components, weights):
            total_sum += current_component * current_weight

        return total_sum

    def compute_average_centroid_distance(self, centroids: torch.Tensor):
        '''
        Computes the average distance between centroids

        :param centroids: (centroids_count, space_dimensions) tensor with centroids
        :return: Average L2 distance between centroids
        '''

        centroids_count = centroids.size(0)

        centroids_1 = centroids.unsqueeze(0)  # (1, centroids_count, space_dimensions)
        centroids_2 = centroids.unsqueeze(1)  # (centroids_count, 1, space_dimensions)
        centroids_sum = (centroids_1 - centroids_2).pow(2).sum(2).sqrt().sum()
        average_centroid_distance = centroids_sum / (centroids_count * (centroids_count - 1))

        return average_centroid_distance

    def plot_action_direction_space(self, estimated_action_centroids: torch.Tensor, action_directions_distribution: torch.Tensor,
                                    action_logits: torch.Tensor) -> Image:
        '''
        Saves and returns a plot of the action direction space
        :param estimated_action_centroids: estimated action centroids in the format required by TensorDisplayer
        :param action_directions_distribution: distribution of action directions in the format required by TensorDisplayer
        :param action_logits: action logits with the space required by TensorDisplayer. Automatically converted to
                              probabilities before being passed to TensorDisplayer
        :return: Image with the plot
        '''

        with torch.no_grad():
            action_probabilities = torch.softmax(action_logits, dim=-1)

        plot_filename = os.path.join(self.config["logging"]["output_images_directory"], f"action_direction_space_{self.global_step}.png")
        TensorDisplayer.show_action_directions(estimated_action_centroids, action_directions_distribution, action_probabilities, plot_filename)
        return Image.open(plot_filename)

    def plot_action_states(self, action_states: torch.Tensor, action_logits: torch.Tensor) -> Image:
        '''
        Saves and returns a plot of the action state trajectories

        :param action_states: (bs, observations_count, action_space_dimension) action state trajectories
                              or (bs, observations_count, 2, action_space_dimension) action state distribution trajectories
        :param action_logits: action logits with the space required by TensorDisplayer. Automatically converted to
                              probabilities before being passed to TensorDisplayer
        :return: Image with the plot
        '''

        with torch.no_grad():
            action_probabilities = torch.softmax(action_logits, dim=-1)

        plot_filename = os.path.join(self.config["logging"]["output_images_directory"], f"action_state_trajectories_{self.global_step}.png")
        TensorDisplayer.show_action_states(action_states, action_probabilities, plot_filename)
        return Image.open(plot_filename)

    def compute_losses_pretraining(self, model, batch: Batch, observations_count: int) -> Tuple:
        '''
        Computes losses for the pretraining phase

        :param model: The network model
        :param batch: Batch of data
        :param observations_count: The number of observations in each sequence
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
        '''

        # Ground truth observations to use at the current step
        ground_truth_observations_count = self.get_ground_truth_observations_count()
        # Since the annealing of the ground truth observations to use may produce a number greater than the number of
        # observations in the sequence, we cap it to the maximum value for the current sequence length
        if ground_truth_observations_count >= observations_count:
            ground_truth_observations_count = observations_count - 1

        # Gumbel temperature to use at the current step
        gumbel_temperture = self.get_gumbel_temperature()

        # Computes forward and losses for the plain batch
        batch_tuple = batch.to_tuple()
        results = model(batch_tuple, pretraining=True, gumbel_temperature=gumbel_temperture)
        reconstructed_observations, multiresolution_reconstructed_observations, reconstructed_states, states, reconstructed_hidden_states, hidden_states, selected_actions, action_logits,\
        action_samples, attention, action_directions_distribution, sampled_action_directions, \
        action_states_distribution, sampled_action_states, action_variations, \
        reconstructed_action_logits, \
        reconstructed_action_directions_distribution, reconstructed_sampled_action_directions, \
        reconstructed_action_states_distribution, reconstructed_sampled_action_states, \
        *other_results = results

        estimated_action_centroids = model.module.centroid_estimator.get_estimated_centroids()

        # Computes the weights mask
        weights_mask = None
        if self.config["training"]["use_motion_weights"]:
            ground_truth_observations = batch_tuple[0]
            weights_mask = self.weight_mask_calculator.compute_weight_mask(ground_truth_observations, reconstructed_observations)

        perceptual_loss_lambda = self.config["training"]["loss_weights"]["perceptual_loss_lambda_pretraining"]

        loss_info_reconstruction = {}

        # Computes perceptual and observation reconstruction losses averaged over all resolutions
        resolutions_count = len(multiresolution_reconstructed_observations)
        perceptual_loss = torch.zeros((1,), dtype=float).cuda()
        perceptual_loss_term = torch.zeros((1,), dtype=float).cuda()
        observations_rec_loss = torch.zeros((1,), dtype=float).cuda()

        for resolution_idx, current_reconstructed_observations in enumerate(multiresolution_reconstructed_observations):

            current_perceptual_loss, current_perceptual_loss_components = self.perceptual_loss(batch.observations, current_reconstructed_observations, weights_mask)
            current_perceptual_loss_term = self.sum_loss_components(current_perceptual_loss_components, perceptual_loss_lambda)
            current_observations_rec_loss = self.observations_loss(batch.observations, current_reconstructed_observations, weights_mask)

            perceptual_loss += current_perceptual_loss
            perceptual_loss_term += current_perceptual_loss_term
            observations_rec_loss += current_observations_rec_loss

            loss_info_reconstruction[f'perceptual_loss_r{resolution_idx}'] = current_perceptual_loss.item()
            loss_info_reconstruction[f'observations_rec_loss_r{resolution_idx}'] = current_observations_rec_loss.item()
            for layer_idx, component in enumerate(current_perceptual_loss_components):
                loss_info_reconstruction[f'perceptual_loss_r{resolution_idx}_l{layer_idx}'] = current_perceptual_loss_components[layer_idx].item()

        perceptual_loss /= resolutions_count
        perceptual_loss_term /= resolutions_count
        observations_rec_loss /= resolutions_count

        states_rec_loss = self.states_loss(states.detach(), reconstructed_states)
        hidden_states_rec_loss = self.hidden_states_loss(hidden_states, reconstructed_hidden_states.detach()) # Avoids gradient backpropagation from dynamics to representation network
        entropy_loss = self.entropy_loss(action_logits)
        action_directions_kl_divergence_loss = self.action_directions_kl_gaussian_divergence_loss(action_directions_distribution)
        action_mutual_information_loss = self.mutual_information_loss(torch.softmax(action_logits, dim=-1),
                                                                      torch.softmax(reconstructed_action_logits, dim=-1),
                                                                      lamb=self.action_mutual_infromation_entropy_lambda)
        action_state_distribution_kl_loss = self.action_state_distribution_kl(reconstructed_action_states_distribution, action_states_distribution.detach())  # The reconstructed must get closer to the true ones, not the contrary


        # Additional debug information not used for backpropagation
        with torch.no_grad():
            samples_entropy = self.samples_entropy_loss(action_samples)
            action_ditribution_entropy = self.action_distribution_entropy(action_samples.mean(dim=(0, 1)).unsqueeze(dim=0))
            states_magnitude = torch.mean(torch.abs(states)).item()
            hidden_states_magnitude = torch.mean(torch.abs(hidden_states)).item()
            action_directions_mean_magnitude = torch.mean(torch.abs(action_directions_distribution[:, :, 0])).item()  # Compute magnitude of the mean
            action_directions_variance_magnitude = torch.mean(torch.abs(action_directions_distribution[:, :, 1])).item()  # Compute magnitude of the variance
            reconstructed_action_directions_mean_magnitude = torch.mean(torch.abs(reconstructed_action_directions_distribution[:, :, 0])).item()  # Compute magnitude of the mean
            reconstructed_action_directions_variance_magnitude = torch.mean(torch.abs(reconstructed_action_directions_distribution[:, :, 1])).item()  # Compute magnitude of the variance
            action_directions_reconstruction_error = torch.mean((reconstructed_action_directions_distribution[:, :, 0] - action_directions_distribution[:, :, 0]).pow(2)).item()  # Compute differences of the mean
            reconstructed_action_directions_kl_divergence_loss = self.action_directions_kl_gaussian_divergence_loss(reconstructed_action_directions_distribution)
            centroids_mean_magnitude = torch.mean(torch.abs(estimated_action_centroids)).item()
            average_centroids_distance = self.compute_average_centroid_distance(estimated_action_centroids).item()
            average_action_variations_norm_l2 = action_variations.pow(2).sum(-1).sqrt().mean().item()
            action_variations_mean = action_variations.mean().item()

        # Computes the total loss
        total_loss = self.config["training"]["loss_weights"]["reconstruction_loss_lambda_pretraining"] * observations_rec_loss + \
                     perceptual_loss_term + \
                     self.config["training"]["loss_weights"]["hidden_states_rec_lambda_pretraining"] * hidden_states_rec_loss + \
                     self.config["training"]["loss_weights"]["states_rec_lambda_pretraining"] * states_rec_loss + \
                     self.config["training"]["loss_weights"]["entropy_lambda_pretraining"] * entropy_loss + \
                     self.config["training"]["loss_weights"]["action_directions_kl_lambda_pretraining"] * action_directions_kl_divergence_loss + \
                     self.config["training"]["loss_weights"]["action_mutual_information_lambda_pretraining"] * action_mutual_information_loss + \
                     self.config["training"]["loss_weights"]["action_state_distribution_kl_lambda_pretraining"] * action_state_distribution_kl_loss

        # Computes loss information
        loss_info = {
            "loss_component_observations_rec": self.config["training"]["loss_weights"]["reconstruction_loss_lambda_pretraining"] * observations_rec_loss.item(),
            "loss_component_perceptual_loss": perceptual_loss_term.item(),
            "loss_component_hidden_states_rec": self.config["training"]["loss_weights"]["hidden_states_rec_lambda_pretraining"] * hidden_states_rec_loss.item(),
            "loss_component_states_rec": self.config["training"]["loss_weights"]["states_rec_lambda_pretraining"] * states_rec_loss .item(),
            "loss_component_entropy": self.config["training"]["loss_weights"]["entropy_lambda_pretraining"] * entropy_loss.item(),
            "loss_component_action_directions_kl_divergence": self.config["training"]["loss_weights"]["action_directions_kl_lambda_pretraining"] * action_directions_kl_divergence_loss.item(),
            "loss_component_action_mutual_information": self.config["training"]["loss_weights"]["action_mutual_information_lambda_pretraining"] * action_mutual_information_loss.item(),
            "loss_component_action_state_distribution_kl": self.config["training"]["loss_weights"]["action_state_distribution_kl_lambda_pretraining"] * action_state_distribution_kl_loss.item(),
            "avg_observations_rec_loss": observations_rec_loss.item(),
            "avg_perceptual_loss": perceptual_loss.item(),
            "states_rec_loss": states_rec_loss.item(),
            "hidden_states_rec_loss": hidden_states_rec_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "samples_entropy": samples_entropy.item(),
            "action_distribution_entropy": action_ditribution_entropy.item(),
            "states_magnitude": states_magnitude,
            "hidden_states_magnitude": hidden_states_magnitude,
            "action_directions_mean_magnitude": action_directions_mean_magnitude,
            "action_directions_variance_magnitude": action_directions_variance_magnitude,
            "reconstructed_action_directions_mean_magnitude": reconstructed_action_directions_mean_magnitude,
            "reconstructed_action_directions_variance_magnitude": reconstructed_action_directions_variance_magnitude,
            "action_directions_reconstruction_error": action_directions_reconstruction_error,
            "action_directions_kl_loss": action_directions_kl_divergence_loss.item(),
            "centroids_mean_magnitude": centroids_mean_magnitude,
            "average_centroids_distance": average_centroids_distance,
            "average_action_variations_norm_l2": average_action_variations_norm_l2,
            "action_variations_mean": action_variations_mean,
            "reconstructed_action_directions_kl_loss": reconstructed_action_directions_kl_divergence_loss.item(),
            "action_mutual_information_loss": action_mutual_information_loss.item(),
            "action_state_distribution_kl_loss": action_state_distribution_kl_loss.item(),
            "ground_truth_observations": ground_truth_observations_count,
            "gumbel_temperature": gumbel_temperture,
            "observations_count": observations_count,
            }
        loss_info = dict(loss_info, **loss_info_reconstruction)

        additional_info = {

        }

        # Plots the action direction space at regular intervals
        if self.global_step % self.config["training"]["action_direction_plotting_freq"] == 0:
            image = self.plot_action_direction_space(estimated_action_centroids, action_directions_distribution, action_logits)
            additional_info["action_direction_space"] = wandb.Image(image)
            image = self.plot_action_states(sampled_action_states, action_logits)
            additional_info["action_state_trajectories"] = wandb.Image(image)

        return total_loss, loss_info, additional_info

    def compute_losses(self, model, batch: Batch, observations_count: int) -> Tuple:
        '''
        Computes losses using the full model

        :param model: The network model
        :param batch: Batch of data
        :param observations_count: The number of observations in each sequence
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
        '''

        # Ground truth observations to use at the current step
        ground_truth_observations_count = self.get_ground_truth_observations_count()

        # Since the annealing of the ground truth observations to use may produce a number greater than the number of
        # observations in the sequence, we cap it to the maximum value for the current sequence length
        if ground_truth_observations_count >= observations_count:
            ground_truth_observations_count = observations_count - 1

        # Gumbel temperature to use at the current step
        gumbel_temperature = self.get_gumbel_temperature()

        # Computes forward and losses for the plain batch
        batch_tuple = batch.to_tuple()
        results = model(batch_tuple, ground_truth_observations_count, gumbel_temperature=gumbel_temperature)
        reconstructed_observations, multiresolution_reconstructed_observations, reconstructed_states, states, hidden_states, selected_actions, action_logits, action_samples, \
        attention, reconstructed_attention, action_directions_distribution, sampled_action_directions, \
        action_states_distribution, sampled_action_states, action_variations,\
        reconstructed_action_logits, \
        reconstructed_action_directions_distribution, reconstructed_sampled_action_directions, \
        reconstructed_action_states_distribution, reconstructed_sampled_action_states,  *other_results = results

        estimated_action_centroids = model.module.centroid_estimator.get_estimated_centroids()

        # Computes the weights mask
        weights_mask = None
        if self.config["training"]["use_motion_weights"]:
            ground_truth_observations = batch_tuple[0]
            weights_mask = self.weight_mask_calculator.compute_weight_mask(ground_truth_observations, reconstructed_observations)

        perceptual_loss_lambda = self.config["training"]["loss_weights"]["perceptual_loss_lambda"]
        loss_info_reconstruction = {}

        # Computes perceptual and observation reconstruction losses averaged over all resolutions
        resolutions_count = len(multiresolution_reconstructed_observations)
        perceptual_loss = torch.zeros((1,), dtype=float).cuda()
        perceptual_loss_term = torch.zeros((1,), dtype=float).cuda()
        observations_rec_loss = torch.zeros((1,), dtype=float).cuda()
        for resolution_idx, current_reconstructed_observations in enumerate(multiresolution_reconstructed_observations):
            current_perceptual_loss, current_perceptual_loss_components = self.perceptual_loss(batch.observations, current_reconstructed_observations, weights_mask)
            current_perceptual_loss_term = self.sum_loss_components(current_perceptual_loss_components, perceptual_loss_lambda)
            current_observations_rec_loss = self.observations_loss(batch.observations, current_reconstructed_observations, weights_mask)

            perceptual_loss += current_perceptual_loss
            perceptual_loss_term += current_perceptual_loss_term
            observations_rec_loss += current_observations_rec_loss

            loss_info_reconstruction[f'perceptual_loss_r{resolution_idx}'] = current_perceptual_loss.item()
            loss_info_reconstruction[f'observations_rec_loss_r{resolution_idx}'] = current_observations_rec_loss.item()
            for layer_idx, component in enumerate(current_perceptual_loss_components):
                loss_info_reconstruction[f'perceptual_loss_r{resolution_idx}_l{layer_idx}'] = current_perceptual_loss_components[layer_idx].item()

        perceptual_loss /= resolutions_count
        perceptual_loss_term /= resolutions_count
        observations_rec_loss /= resolutions_count

        states_rec_loss = self.states_loss(states.detach(), reconstructed_states)
        entropy_loss = self.entropy_loss(action_logits)
        action_directions_kl_divergence_loss = self.action_directions_kl_gaussian_divergence_loss(action_directions_distribution)
        action_mutual_information_loss = self.mutual_information_loss(torch.softmax(action_logits, dim=-1),
                                                                      torch.softmax(reconstructed_action_logits, dim=-1),
                                                                      lamb=self.action_mutual_infromation_entropy_lambda)
        action_state_distribution_kl_loss = self.action_state_distribution_kl(reconstructed_action_states_distribution, action_states_distribution.detach())   # The reconstructed must get closer to the true ones, not the contrary

        # Additional debug information not used for backpropagation
        with torch.no_grad():
            samples_entropy = self.samples_entropy_loss(action_samples)
            action_distribution_entropy = self.action_distribution_entropy(action_samples.mean(dim=(0, 1)).unsqueeze(dim=0))
            states_magnitude = torch.mean(torch.abs(states)).item()
            hidden_states_magnitude = torch.mean(torch.abs(hidden_states)).item()
            action_directions_mean_magnitude = torch.mean(torch.abs(action_directions_distribution[:, :, 0])).item()  # Compute magnitude of the mean
            action_directions_variance_magnitude = torch.mean(torch.abs(action_directions_distribution[:, :, 1])).item()  # Compute magnitude of the variance
            reconstructed_action_directions_mean_magnitude = torch.mean(torch.abs(reconstructed_action_directions_distribution[:, :, 0])).item() # Compute magnitude of the mean
            reconstructed_action_directions_variance_magnitude = torch.mean(torch.abs(reconstructed_action_directions_distribution[:, :, 1])).item()  # Compute magnitude of the variance
            action_directions_reconstruction_error = torch.mean((reconstructed_action_directions_distribution[:, :, 0] - action_directions_distribution[:, :, 0]).pow(2)).item() # Compute differences of the mean
            reconstructed_action_directions_kl_divergence_loss = self.action_directions_kl_gaussian_divergence_loss(reconstructed_action_directions_distribution)
            centroids_mean_magnitude = torch.mean(torch.abs(estimated_action_centroids)).item()
            average_centroids_distance = self.compute_average_centroid_distance(estimated_action_centroids).item()
            average_action_variations_norm_l2 = action_variations.pow(2).sum(-1).sqrt().mean().item()
            action_variations_mean = action_variations.mean().item()

        # Computes the total loss
        total_loss = self.config["training"]["loss_weights"]["reconstruction_loss_lambda"] * observations_rec_loss + \
                     perceptual_loss_term + \
                     self.config["training"]["loss_weights"]["states_rec_lambda"] * states_rec_loss + \
                     self.config["training"]["loss_weights"]["entropy_lambda"] * entropy_loss + \
                     self.config["training"]["loss_weights"]["action_directions_kl_lambda"] * action_directions_kl_divergence_loss + \
                     self.config["training"]["loss_weights"]["action_mutual_information_lambda"] * action_mutual_information_loss + \
                     self.config["training"]["loss_weights"]["action_state_distribution_kl_lambda"] * action_state_distribution_kl_loss

        # Computes loss information
        loss_info = {
            "loss_component_observations_rec": self.config["training"]["loss_weights"]["reconstruction_loss_lambda"] * observations_rec_loss.item(),
            "loss_component_perceptual_loss": perceptual_loss_term.item(),
            "loss_component_states_rec": self.config["training"]["loss_weights"]["states_rec_lambda"] * states_rec_loss.item(),
            "loss_component_entropy": self.config["training"]["loss_weights"]["entropy_lambda"] * entropy_loss.item(),
            "loss_component_action_directions_kl_divergence": self.config["training"]["loss_weights"]["action_directions_kl_lambda"] * action_directions_kl_divergence_loss.item(),
            "loss_component_action_mutual_information": self.config["training"]["loss_weights"]["action_mutual_information_lambda"] * action_mutual_information_loss.item(),
            "loss_component_action_state_distribution_kl": self.config["training"]["loss_weights"]["action_state_distribution_kl_lambda"] * action_state_distribution_kl_loss.item(),
            "avg_observations_rec_loss": observations_rec_loss.item(),
            "avg_perceptual_loss": perceptual_loss.item(),
            "states_rec_loss": states_rec_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "samples_entropy": samples_entropy.item(),
            "action_distribution_entropy": action_distribution_entropy.item(),
            "states_magnitude": states_magnitude,
            "hidden_states_magnitude": hidden_states_magnitude,
            "action_directions_mean_magnitude": action_directions_mean_magnitude,
            "action_directions_variance_magnitude": action_directions_variance_magnitude,
            "reconstructed_action_directions_mean_magnitude": reconstructed_action_directions_mean_magnitude,
            "reconstructed_action_directions_variance_magnitude": reconstructed_action_directions_variance_magnitude,
            "action_directions_reconstruction_error": action_directions_reconstruction_error,
            "action_directions_kl_loss": action_directions_kl_divergence_loss.item(),
            "centroids_mean_magnitude": centroids_mean_magnitude,
            "average_centroids_distance": average_centroids_distance,
            "average_action_variations_norm_l2": average_action_variations_norm_l2,
            "action_variations_mean": action_variations_mean,
            "reconstructed_action_directions_kl_loss": reconstructed_action_directions_kl_divergence_loss.item(),
            "action_mutual_information_loss": action_mutual_information_loss.item(),
            "action_state_distribution_kl_loss": action_state_distribution_kl_loss.item(),
            "ground_truth_observations": ground_truth_observations_count,
            "gumbel_temperature": gumbel_temperature,
            "observations_count": observations_count,
            }
        # Concatenates the info dictionaries
        loss_info = dict(loss_info, **loss_info_reconstruction)

        additional_info = {

        }

        # Plots the action direction space at regular intervals
        if self.global_step % self.config["training"]["action_direction_plotting_freq"] == 0:
            image = self.plot_action_direction_space(estimated_action_centroids, action_directions_distribution, action_logits)
            additional_info["action_direction_space"] = wandb.Image(image)
            image = self.plot_action_states(sampled_action_states, action_logits)
            additional_info["action_state_trajectories"] = wandb.Image(image)

        return total_loss, loss_info, additional_info

    def train_epoch(self, model):

        self.logger.print(f'== Train [{self.global_step}] ==')

        # Computes the number of observations to use in the current epoch
        observations_count = self.get_observations_count()
        # Modifies the number of observations to return before instantiating the dataloader
        self.dataset.set_observations_count(observations_count)

        # Number of training steps performed in this epoch
        performed_steps = 0
        for step, batch_group in enumerate(self.dataloader):
            # If the maximum number of training steps per epoch is exceeded, we interrupt the epoch
            if performed_steps > self.config["training"]["max_steps_per_epoch"]:
                break

            self.global_step += 1
            performed_steps += 1

            # If there is a change in the number of observations to use, we interrupt the epoch
            current_observations_count = self.get_observations_count()
            if current_observations_count != observations_count:
                break


            if self.global_step <= self.config["training"]["pretraining_steps"]:
                loss, loss_info, additional_info = self.compute_losses_pretraining(model, batch_group, observations_count)
            else:
                loss, loss_info, additional_info = self.compute_losses(model, batch_group, observations_count)
            # Logs the loss
            loss_info["loss"] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            self.average_meter.add(loss_info)

            if (self.global_step - 1) % 1 == 0:

                self.logger.print(f'step: {self.global_step}/{self.config["training"]["max_steps"]}', end=" ")

                average_values = {description: self.average_meter.pop(description) for description in loss_info}
                for description, value in average_values.items():
                    self.logger.print("{}:{:.3f}".format(description, value), end=" ")

                current_lr = self._get_current_lr()
                self.logger.print('lr: %.4f' % (current_lr))

                if (self.global_step - 1) % 10 == 0:
                    wandb = self.logger.get_wandb()
                    logged_map = {"train/" + description: item for description, item in average_values.items()}
                    logged_map["step"] = self.global_step
                    logged_map["train/lr"] = current_lr
                    wandb.log(logged_map, step=self.global_step)
                    additional_info["step"] = self.global_step
                    wandb.log(additional_info, step=self.global_step)


def trainer(config, model, dataset, logger):
    return Trainer(config, model, dataset, logger)