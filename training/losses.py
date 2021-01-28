from typing import Tuple, List
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.vgg import Vgg19
from utils.memory_displayer import MemoryDisplayer
from utils.tensor_folder import TensorFolder


class StatesLoss:

    def __init__(self):
        self.loss = nn.MSELoss()  # Manually account for averaging

    def __call__(self, states: torch.Tensor, reconstructed_states: torch.Tensor):
        '''

        :param states: (bs, observations_count, state_h, state_w) tensor with observed states
        :param reconstructed_states: (bs, observations_count, state_h, state_w) tensor with reconstructed states
        :return:
        '''

        return self.loss(states, reconstructed_states)


class HiddenStatesLoss:

    def __init__(self):
        self.loss = nn.MSELoss()  # Manually account for averaging

    def __call__(self, hidden_states: torch.Tensor, reconstructed_hidden_states: torch.Tensor):
        '''

        :param hidden_states: (bs, observations_count, state_h, state_w) tensor with observed states
        :param reconstructed_hidden_states: (bs, observations_count, state_h, state_w) tensor with reconstructed states
        :return:
        '''

        sequence_length = hidden_states.size(1)
        reconstructed_sequence_length = reconstructed_hidden_states.size(1)

        # If the length of the sequences differ, the first reconstructed frame if probably not available
        # We remove it from the batch
        if reconstructed_sequence_length != sequence_length:
            if reconstructed_sequence_length - 1 != sequence_length:
                raise Exception(f"Received an input batch with sequence length {sequence_length}, but got a reconstructed batch of {reconstructed_sequence_length}")
            reconstructed_hidden_states = reconstructed_hidden_states[:, 1:]

        return self.loss(hidden_states, reconstructed_hidden_states)


class ObservationsLoss:

    def __init__(self):
        self.loss = nn.L1Loss()

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, weight_mask=None):
        '''

        :param observations: (bs, observations_count, 3*observation_stacking, h, w) ground truth observations. Rescaled if needed
        :param reconstructed_observations: (bs, observations_count|observations_count-1, 3, height, width) tensor with reconstructed frames
        :param weight_mask: (bs, observations_count, 1, h, w) tensor weights to assign to each spatial position for loss computation. Rescaled if needed
        :return:
        '''


        observations = observations[:, :, :3] # For each observation extract only the current frame and not the past ones

        sequence_length = observations.size(1)
        reconstructed_sequence_length = reconstructed_observations.size(1)

        original_observation_height = observations.size(3)
        original_observation_width = observations.size(4)
        height = reconstructed_observations.size(3)
        width = reconstructed_observations.size(4)

        # If the length of the sequences differ, the first reconstructed frame if probably not available
        # We remove it from the batch
        if reconstructed_sequence_length != sequence_length:
            if reconstructed_sequence_length != sequence_length - 1:
                raise Exception(f"Received an input batch with sequence length {sequence_length}, but got a reconstructed batch of {reconstructed_sequence_length}")
            observations = observations[:, 1:]
            sequence_length -= 1

        flattened_observations = TensorFolder.flatten(observations)
        flattened_reconstructed_observations = TensorFolder.flatten(reconstructed_observations)

        flattened_observations = F.interpolate(flattened_observations, (height, width), mode='bilinear')

        # If a weight mask is specified use weighted loss computation
        if weight_mask is not None:
            weight_mask_length = weight_mask.size(1)
            # Ensures all the input tensors have the same sequence length
            if weight_mask_length != reconstructed_sequence_length:
                if reconstructed_sequence_length != weight_mask_length - 1:
                    raise Exception(f"Received a reconstructed sequence with length {reconstructed_sequence_length}, but got a weight mast of length {weight_mask_length}")
                weight_mask = weight_mask[:, 1:]
            observations_channels = observations.size(2)
            assert(observations_channels == 3)

            flattened_weight_mask = TensorFolder.fold(weight_mask)
            flattened_weight_mask = F.interpolate(flattened_weight_mask, (height, width), mode='bilinear')
            unreduced_loss = torch.abs(flattened_observations - flattened_reconstructed_observations)
            # Computes the weighted sum of the loss using the weight mask as weights
            # Computes the loss such that each frame has the same relative importance, independently of its mask
            # The weights change only the relative importance between pixels of a single image
            unreduced_loss = unreduced_loss * flattened_weight_mask
            loss = unreduced_loss.sum(dim=(2, 3))
            loss = loss / (weight_mask.sum(dim=(2, 3)) * observations_channels) # Since weight mask is broadcasted in the channel
                                                                                      # directions we need to multiply per the number of channels
            return loss.mean()

        # Otherwise use unweighted loss computation
        return self.loss(flattened_observations, flattened_reconstructed_observations)


class KLDivergence:

    def __init__(self):
        pass

    def __call__(self, input_logits: torch.Tensor, target_logits: torch.Tensor):
        '''
        Computes the KL divergence between the logits of two probability distributions

        :param input_logits: (bs, observations_count, distribution_cardinality) tensor of network output logits
        :param target_logits: (bs, observations_count, distribution_cardinality) tensor of network output logits
        :return: KL distance between the two probability distributions
        '''

        actions_count = input_logits.size(-1)
        flattened_input_logits = input_logits.reshape(-1, actions_count)
        flattened_target_logits = target_logits.reshape(-1, actions_count)

        input_log_probabilities = F.log_softmax(flattened_input_logits, dim=1)
        target_probabilities = F.softmax(flattened_target_logits, dim=1)

        # Second probabilities must not be given in log format
        return F.kl_div(input_log_probabilities, target_probabilities, reduction="batchmean") # Match math definition


class KLGaussianDivergenceLoss:

    def __init__(self):
        pass

    def __call__(self, distribution_parameters: torch.Tensor):
        '''
        Computes the KL divergence between the given distribution and the N(0, 1) distribution

        :param distribution_parameters: (..., 2, space_dimension) tensor with distribution mean and variances
        :return: KL distance between the given distribution and the N(0, 1) distribution
        '''

        space_dimension = distribution_parameters.size(-1)
        distribution_parameters = distribution_parameters.view(-1, 2, space_dimension)
        mean = distribution_parameters[:, 0]
        variance = distribution_parameters[:, 1]
        log_variance = torch.log(variance)

        kl = 1 + log_variance - mean.pow(2) - variance  # (bs, space_dimension)
        kl = kl.sum(dim=-1)  # Sums across the space dimension
        kl = -0.5 * kl.mean()  # Averages across the batch dimension

        return kl


class KLGeneralGaussianDivergenceLoss:

    def __init__(self):
        pass

    def __call__(self, distribution_parameters: torch.Tensor, reference_distribution_parameters: torch.Tensor, eps=0.05):
        '''
        Computes the KL divergence between two given distributions

        :param distribution_parameters: (..., 2, space_dimension) tensor with distribution mean and variances
        :param reference_distribution_parameters: (..., 2, space_dimension) tensor with distribution mean and variances
        :return: KL distance between distribution and the reference distribution
        '''

        space_dimension = distribution_parameters.size(-1)
        distribution_parameters = distribution_parameters.view(-1, 2, space_dimension)
        reference_distribution_parameters = reference_distribution_parameters.view(-1, 2, space_dimension)

        mean = distribution_parameters[:, 0]
        variance = distribution_parameters[:, 1].detach()  # Do not backpropagate through variance
        log_variance = torch.log(variance)

        reference_mean = reference_distribution_parameters[:, 0]
        reference_variance = reference_distribution_parameters[:, 1].detach()  # Do not backpropagate through variance
        reference_log_variance = torch.log(reference_variance)

        variance = torch.clamp(variance, min=eps)
        reference_variance = torch.clamp(reference_variance, min=eps)

        variance_ratio = variance / reference_variance
        mus_term = (reference_mean - mean).pow(2) / reference_variance

        kl = reference_log_variance - log_variance - 1 + variance_ratio + mus_term  # (bs, space_dimension)

        kl = kl.sum(dim=-1)  # Sums across the space dimension
        kl = 0.5 * kl.mean()  # Averages across the batch dimension

        return kl


class FixedMatrixEstimator(nn.Module):

    def __init__(self, rows, columns, initial_alpha=0.2, initial_value=None):
        '''
        Initializes the joint probability estimator for a (rows, columns) matrix with the given fixed alpha factor
        :param rows, columns: Dimension of the probability matrix to estimate
        :param initial_alpha: Value to use assign for alpha
        '''
        super(FixedMatrixEstimator, self).__init__()

        self.alpha = initial_alpha

        # Initializes the joint matrix as a uniform, independent distribution. Does not allow backpropagation to this parameter
        if initial_value is None:
            initial_value = torch.tensor([[1.0 / (rows * columns)] * columns] * rows, dtype=torch.float32)
        self.estimated_matrix = nn.Parameter(initial_value, requires_grad=False)

    def forward(self, latest_probability_matrix):
        return_matrix = self.estimated_matrix * (1 - self.alpha) + latest_probability_matrix * self.alpha

        # The estimated matrix must be detached from the backpropagation graph to avoid exhaustion of GPU memory
        self.estimated_matrix.data = return_matrix.detach()

        return return_matrix


class MutualInformationLoss(nn.Module):

    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def compute_joint_probability_matrix(self, distribution_1: torch.Tensor,
                                         distribution_2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the joint probability matrix

        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :return: (dim, dim) tensor with joint probability matrix
        '''

        # Flattens the distributions
        dim = distribution_1.size(-1)
        assert (distribution_2.size(-1) == dim)
        distribution_1 = distribution_1.view(-1, dim)
        distribution_2 = distribution_2.view(-1, dim)

        batch_size = distribution_1.size(0)
        assert (distribution_2.size(0) == batch_size)

        p_i_j = distribution_1.unsqueeze(2) * distribution_2.unsqueeze(1)  # (batch_size, dim, dim)
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def __call__(self, distribution_1: torch.Tensor, distribution_2: torch.Tensor, lamb=1.0,
                 eps=sys.float_info.epsilon) -> torch.Tensor:
        '''
        Computes the mutual information loss for a joint probability matrix
        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :param lamb: lambda parameter to change the importance of entropy in the loss
        :param eps: small constant for numerical stability
        :return: mutual information loss for the given joint probability matrix
        '''

        # Computes the joint probability matrix
        joint_probability_matrix = self.compute_joint_probability_matrix(distribution_1, distribution_2)
        rows, columns = joint_probability_matrix.size()

        # Computes the marginals
        marginal_rows = joint_probability_matrix.sum(dim=1).view(rows, 1).expand(rows, columns)
        marginal_columns = joint_probability_matrix.sum(dim=0).view(1, columns).expand(rows,
                                                                                       columns)  # but should be same, symmetric

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        joint_probability_matrix[(joint_probability_matrix < eps).data] = eps
        marginal_rows = marginal_rows.clone()
        marginal_columns = marginal_columns.clone()
        marginal_rows[(marginal_rows < eps).data] = eps
        marginal_columns[(marginal_columns < eps).data] = eps

        mutual_information = joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                                         - lamb * torch.log(marginal_rows) \
                                                         - lamb * torch.log(marginal_columns))

        mutual_information = mutual_information.sum()

        return -1 * mutual_information


class SmoothMutualInformationLoss(MutualInformationLoss):
    '''
    Mutual information loss with smooth joint probability matrix estimation
    '''

    def __init__(self, config):
        '''
        Creates the loss according to the specified configuration
        :param config: The configuration
        '''

        super(SmoothMutualInformationLoss, self).__init__()

        self.actions_count = config["data"]["actions_count"]
        self.mi_estimation_alpha = config["training"]["mutual_information_estimation_alpha"]
        self.matrix_estimator = FixedMatrixEstimator(self.actions_count, self.actions_count, self.mi_estimation_alpha)

    def compute_joint_probability_matrix(self, distribution_1: torch.Tensor,
                                         distribution_2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the joint probability matrix

        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :return: (dim, dim) tensor with joint probability matrix
        '''

        # Compute the joint probability matrix as before
        current_joint_probability_matrix = super(SmoothMutualInformationLoss, self).compute_joint_probability_matrix(distribution_1, distribution_2)
        # Smooth the joint probability matrix with the estimator
        smoothed_joint_probability_matrix = self.matrix_estimator(current_joint_probability_matrix)
        return smoothed_joint_probability_matrix


class EntropyLogitLoss:

    def __init__(self):
        pass

    def __call__(self, logits: torch.Tensor):
        '''
        Computes the entropy of the passed logits
        :param logits: (..., classes_counts) tensor
        :return: entropy over the last dimension averaged on each sample
        '''

        classes_count = logits.size(-1)
        flat_logits = logits.reshape((-1, classes_count))
        samples_count = flat_logits.size(0)

        entropy = -1 * torch.sum(F.softmax(flat_logits, dim=1) * F.log_softmax(flat_logits, dim=1)) / samples_count
        return entropy


class EntropyProbabilityLoss:

    def __init__(self):
        pass

    def __call__(self, probabilities: torch.Tensor):
        '''
        Computes the entropy of the passed probabilities
        :param probabilities: (..., classes_counts) tensor with probabilitiy distribution over the classes
        :return: entropy over the last dimension averaged on each sample
        '''

        classes_count = probabilities.size(-1)
        flat_probabilities = probabilities.reshape((-1, classes_count))
        samples_count = flat_probabilities.size(0)

        entropy = -1 * torch.sum(flat_probabilities * torch.log(flat_probabilities)) / samples_count
        return entropy


class ParallelPerceptualLoss:

    def __init__(self):

        self.perceptual_loss = UnmeanedPerceptualLoss()
        self.perceptual_loss = nn.DataParallel(self.perceptual_loss).cuda()

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, weight_mask=None):
        total_loss, individual_losses = self.perceptual_loss(observations, reconstructed_observations, weight_mask)

        meaned_individual_losses = [current_loss.mean() for current_loss in individual_losses]
        return total_loss.mean(), meaned_individual_losses


class UnmeanedPerceptualLoss(nn.Module):

    def __init__(self):

        super(UnmeanedPerceptualLoss, self).__init__()

        self.vgg = Vgg19()
        self.vgg = self.vgg

    def forward(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, weight_mask=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Computes the perceptual loss between the sets of observations

        :param observations: (bs, observations_count, 3*observation_stacking, h, w) ground truth observations. Rescaled if needed
        :param reconstructed_observations: (bs, observations_count|observations_count-1, 3, height, width) tensor with reconstructed observations
        :param weight_mask: (bs, observations_count, 1, h, w) tensor weights to assign to each spatial position for loss computation. Rescaled if needed

        :return: total_loss, individual_losses Perceptual loss between ground truth and reconstructed observations. Both the total loss and the loss
                 for each vgg feature level are returned. Losses have a batch size dimension
        '''

        ground_truth_observations = observations[:, :, :3] # For each observation extract only the current frame and not the past ones

        sequence_length = ground_truth_observations.size(1)
        reconstructed_sequence_length = reconstructed_observations.size(1)

        original_observation_height = ground_truth_observations.size(3)
        original_observation_width = ground_truth_observations.size(4)
        height = reconstructed_observations.size(3)
        width = reconstructed_observations.size(4)

        # If the length of the sequences differ, the first reconstructed frame if probably not available
        # We remove it from the batch
        if reconstructed_sequence_length != sequence_length:
            if reconstructed_sequence_length != sequence_length - 1:
                raise Exception(f"Received an input batch with sequence length {sequence_length}, but got a reconstructed batch of {reconstructed_sequence_length}")
            ground_truth_observations = ground_truth_observations[:, 1:]


        # Check weight mask length
        if weight_mask is not None:
            weight_mask_length = weight_mask.size(1)
            # Ensures all the input tensors have the same sequence length
            if weight_mask_length != reconstructed_sequence_length:
                if reconstructed_sequence_length != weight_mask_length - 1:
                    raise Exception(f"Received a reconstructed sequence with length {reconstructed_sequence_length}, but got a weight mast of length {weight_mask_length}")
                weight_mask = weight_mask[:, 1:]
            weight_height = weight_mask.size(3)
            weight_width = weight_mask.size(4)
            flat_weight_shape = (-1, 1, weight_height, weight_width)
            flattened_weight_mask = weight_mask.reshape(flat_weight_shape)

        flattened_ground_truth_observations = TensorFolder.flatten(ground_truth_observations)
        flattened_reconstructed_observations = TensorFolder.flatten(reconstructed_observations)

        # Resizes to the resolution of the reconstructed observations if needed
        if original_observation_width != width or original_observation_height != height:
            flattened_ground_truth_observations = F.interpolate(flattened_ground_truth_observations, (height, width), mode='bilinear')

        # Computes vgg features. Do not build the computational graph for the ground truth observations
        with torch.no_grad():
            ground_truth_vgg_features = self.vgg(flattened_ground_truth_observations.detach())
        reconstructed_vgg_features = self.vgg(flattened_reconstructed_observations)


        total_loss = None
        single_losses = []
        # Computes the perceptual loss
        for current_ground_truth_feature, current_reconstructed_feature in zip(ground_truth_vgg_features, reconstructed_vgg_features):

            # Compute unweighted loss
            if weight_mask is None:
                current_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature).mean(dim=[1, 2, 3]) # Detach signals to not backpropagate through the ground truth branch
            # Compute loss scaled by weights
            else:
                current_feature_channels = current_ground_truth_feature.size(1)
                current_feature_height = current_ground_truth_feature.size(2)
                current_feature_width = current_ground_truth_feature.size(3)

                # Resize the weight mask
                scaled_weight_masks = F.interpolate(flattened_weight_mask, size=(current_feature_height, current_feature_width), mode='bilinear', align_corners=False)

                unreduced_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature)
                # Computes the weighted sum of the loss using the weight mask as weights
                # Computes the loss such that each image has the same relative importance
                # Only the relative importance between positions in the same frame is modified
                unreduced_loss = unreduced_loss * scaled_weight_masks
                current_loss = unreduced_loss.sum(dim=(1, 2, 3))
                current_loss = current_loss / (scaled_weight_masks.sum(dim=(1, 2, 3)) * current_feature_channels) # Since weight mask is broadcasted in the channel
                                                                                                                  # directions we need to multiply per the number of channels

            if total_loss is None:
                total_loss = current_loss
            else:
                total_loss += current_loss
            single_losses.append(current_loss)


        return total_loss, single_losses


class PerceptualLoss:

    def __init__(self):

        self.vgg = Vgg19()
        self.vgg = nn.DataParallel(self.vgg)
        self.vgg = self.vgg.cuda()

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, weight_mask=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Computes the perceptual loss between the sets of observations

        :param observations: (bs, observations_count, 3*observation_stacking, h, w) ground truth observations. Rescaled if needed
        :param reconstructed_observations: (bs, observations_count|observations_count-1, 3, height, width) tensor with reconstructed observations
        :param weight_mask: (bs, observations_count, 1, h, w) tensor weights to assign to each spatial position for loss computation. Rescaled if needed

        :return: total_loss, individual_losses Perceptual loss between ground truth and reconstructed observations. Both the total loss and the loss
                 for each vgg feature level are returned
        '''

        ground_truth_observations = observations[:, :, :3] # For each observation extract only the current frame and not the past ones

        sequence_length = ground_truth_observations.size(1)
        reconstructed_sequence_length = reconstructed_observations.size(1)

        original_observation_height = ground_truth_observations.size(3)
        original_observation_width = ground_truth_observations.size(4)
        height = reconstructed_observations.size(3)
        width = reconstructed_observations.size(4)


        # If the length of the sequences differ, the first reconstructed frame if probably not available
        # We remove it from the batch
        if reconstructed_sequence_length != sequence_length:
            if reconstructed_sequence_length != sequence_length - 1:
                raise Exception(f"Received an input batch with sequence length {sequence_length}, but got a reconstructed batch of {reconstructed_sequence_length}")
            ground_truth_observations = ground_truth_observations[:, 1:]

        # Check weight mask length
        if weight_mask is not None:
            weight_mask_length = weight_mask.size(1)
            # Ensures all the input tensors have the same sequence length
            if weight_mask_length != reconstructed_sequence_length:
                if reconstructed_sequence_length != weight_mask_length - 1:
                    raise Exception(f"Received a reconstructed sequence with length {reconstructed_sequence_length}, but got a weight mast of length {weight_mask_length}")
                weight_mask = weight_mask[:, 1:]
            weight_height = weight_mask.size(3)
            weight_width = weight_mask.size(4)
            flat_weight_shape = (-1, 1, weight_height, weight_width)
            flattened_weight_mask = weight_mask.reshape(flat_weight_shape)

        flattened_ground_truth_observations = TensorFolder.flatten(ground_truth_observations)
        flattened_reconstructed_observations = TensorFolder.flatten(reconstructed_observations)

        # Resizes to the resolution of the reconstructed observations if needed
        if original_observation_width != width or original_observation_height != height:
            flattened_ground_truth_observations = F.interpolate(flattened_ground_truth_observations, (height, width), mode='bilinear')

        # Computes vgg features. Do not build the computational graph for the ground truth observations
        with torch.no_grad():
            ground_truth_vgg_features = self.vgg(flattened_ground_truth_observations.detach())

        reconstructed_vgg_features = self.vgg(flattened_reconstructed_observations)

        total_loss = torch.tensor([0.0]).cuda()
        single_losses = []
        # Computes the perceptual loss
        for current_ground_truth_feature, current_reconstructed_feature in zip(ground_truth_vgg_features, reconstructed_vgg_features):

            # Compute unweighted loss
            if weight_mask is None:
                current_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature).mean() # Detach signals to not backpropagate through the ground truth branch
            # Compute loss scaled by weights
            else:
                current_feature_channels = current_ground_truth_feature.size(1)
                current_feature_height = current_ground_truth_feature.size(2)
                current_feature_width = current_ground_truth_feature.size(3)

                # Resize the weight mask
                scaled_weight_masks = F.interpolate(flattened_weight_mask, size=(current_feature_height, current_feature_width), mode='bilinear', align_corners=False)

                unreduced_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature)
                # Computes the weighted sum of the loss using the weight mask as weights
                # Computes the loss such that each image has the same relative importance
                # Only the relative importance between positions in the same frame is modified
                unreduced_loss = unreduced_loss * scaled_weight_masks
                current_loss = unreduced_loss.sum(dim=(1, 2, 3))
                current_loss = current_loss / (scaled_weight_masks.sum(dim=(1, 2, 3)) * current_feature_channels) # Since weight mask is broadcasted in the channel
                                                                                                                  # directions we need to multiply per the number of channels
                current_loss = current_loss.mean()

            total_loss += current_loss
            single_losses.append(current_loss)

        return total_loss, single_losses


class MotionLossWeightMaskCalculator:
    '''
    Class for the creation of weight masks based on motion between frames
    '''

    def __init__(self, weight_bias: float = 0.0):
        '''

        :param weight_bias: constant value to add to each position of the weight masks
        '''

        self.weight_bias = weight_bias

    def compute_weight_mask(self, observations, reconstructed_observations):
        '''
        :param observations: (bs, observations_count, 3*observation_stacking, h, w) ground truth observations
        :param reconstructed_observations: (bs, observations_count|observations_count-1, 3, h, w) tensor with reconstructed frames

        :return: (bs, observations_count, 1, h, w) tensor weights to assign to each spatial position for loss computation
                                                   The first sequence element has constant values in all positions
        '''

        # No gradient must flow through the computation of the weight masks
        observations = observations.detach()
        reconstructed_observations = reconstructed_observations.detach()

        observations = observations[:, :, :3]  # For each observation extract only the current frame and not the past ones

        sequence_length = observations.size(1)
        reconstructed_sequence_length = reconstructed_observations.size(1)

        # If the length of the sequences differ, use the first sequence frame to fill the first missing position in the reconstructed
        if reconstructed_sequence_length != sequence_length:
            if reconstructed_sequence_length != sequence_length - 1:
                raise Exception(
                    f"Received an input batch with sequence length {sequence_length}, but got a reconstructed batch of {reconstructed_sequence_length}")
            reconstructed_observations = torch.cat([observations[:, 0:1], reconstructed_observations], dim=1)

        # Ensure the sequences have the same length
        assert(sequence_length == reconstructed_observations.size(1))

        # Computes corresponding predecessor and successor observations
        successor_observations = observations[:, 1:]
        predecessor_observations = observations[:, :-1]
        successor_reconstructed_observations = reconstructed_observations[:, 1:]
        predecessor_reconstructed_observations = reconstructed_observations[:, :-1]

        weight_mask = torch.abs(successor_observations - predecessor_observations) +\
                      torch.abs(successor_reconstructed_observations - predecessor_reconstructed_observations)

        # Sums the mask along the channel dimension
        assert(weight_mask.size(2) == 3)
        weight_mask = weight_mask.sum(dim=2, keepdim=True)

        # Adds bias to the weights
        weight_mask += self.weight_bias
        # Adds a dummy first sequence element
        weight_mask = torch.cat([torch.ones_like(weight_mask[:, 0:1]), weight_mask], dim=1)
        return weight_mask


class SequenceLossEvaluator:
    ''''
    Generic loss evaluator for sequences
    '''

    def __init__(self, loss):
        '''

        :param loss: Callable acceptinr (bs, sequence_length, ...) tensors and returning the loss tensor
        '''

        self.loss = loss

    def __call__(self, ground_truth_sequence: torch.Tensor, reconstructed_sequence: torch.Tensor) -> Tuple:
        '''
        Evaluates a loss at each position in the given sequence. If the reconstructed sequence is shorter it aligns it
        to the right of the ground_truth_sequence and the loss for the first element is evaluated at 0.

        :param ground_truth_sequence: (bs, sequence_length, ...) tensor with ground truth values
        :param reconstructed_sequence: (bs, sequence_length|sequence_length-1, ...) tensor with reconstructed values
        :return: (avg_loss tensor, (sequence_length) tensor) with the average loss and the loss at each position
                 in the sequence
        '''

        sequence_length = ground_truth_sequence.size(1)
        reconstructed_sequence_length = reconstructed_sequence.size(1)

        current_ground_truth_index = 0
        current_reconstructed_index = 0
        loss_terms = torch.zeros((sequence_length)).cuda()

        # If the length of the sequences differ, the first reconstructed frame if probably not available
        # We remove it from the batch
        if reconstructed_sequence_length != sequence_length:
            if reconstructed_sequence_length != sequence_length - 1:
                raise Exception(f"Received an input batch with sequence length {sequence_length}, but got a reconstructed batch of {reconstructed_sequence_length}")
            loss_terms[current_ground_truth_index] = 0.0
            current_ground_truth_index += 1

        # Computes the loss for each position in the sequence
        while current_ground_truth_index < sequence_length:
            current_loss = self.loss(ground_truth_sequence[:, current_ground_truth_index:current_ground_truth_index + 1],
                                     reconstructed_sequence[:, current_reconstructed_index:current_reconstructed_index + 1])

            # If losses return multiple tensors, the first is the loss term of interest
            if type(current_loss) == tuple:
                current_loss = current_loss[0]

            loss_terms[current_ground_truth_index] = current_loss

            current_ground_truth_index += 1
            current_reconstructed_index += 1

        # If the sequences were of the same length compute the average over all elements
        if current_ground_truth_index == current_reconstructed_index:
            avg_loss = torch.mean(loss_terms)
        # Otherwise the first element is non significant
        else:
            assert(loss_terms[0] == 0.0)
            avg_loss = torch.mean(loss_terms[1:])

        return avg_loss, loss_terms


if __name__ == "__main__":

    distribution = torch.tensor([[[1.0, 1.0], [1.0, 0.005]]])
    reference_distribution = torch.tensor([[[1.0, 1.0], [1.0, 0.05]]])
    general_kl_loss = KLGeneralGaussianDivergenceLoss()

    loss = general_kl_loss(distribution, reference_distribution)
    print(loss)
    loss = general_kl_loss(distribution, reference_distribution, eps=0)
    print(loss)
