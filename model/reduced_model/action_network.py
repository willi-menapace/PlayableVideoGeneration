import torch
import torch.nn as nn

from model.layers.residual_block import ResidualBlock
from utils.tensor_folder import TensorFolder
from utils.tensor_splitter import TensorSplitter


class ActionNetwork(nn.Module):
    '''
    Model that reconstructs the frame associated to a state
    '''

    def __init__(self, config):
        super(ActionNetwork, self).__init__()

        self.config = config
        self.state_features = config["model"]["representation_network"]["state_features"]
        self.actions_count = config["data"]["actions_count"]
        self.action_space_dimension = config["model"]["action_network"]["action_space_dimension"]

        residual_blocks = [
            ResidualBlock(self.state_features, 2 * self.state_features, downsample_factor=2),
            ResidualBlock(2 * self.state_features, 2 * self.state_features, downsample_factor=1),
        ]
        self.residuals = nn.Sequential(*residual_blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Linear layers for the prediction of the posterior parameters in the action space
        self.mean_fc = nn.Linear(2 * self.state_features, self.action_space_dimension)
        self.variance_fc = nn.Linear(2 * self.state_features, self.action_space_dimension)

        # Classifies the vector
        self.final_fc = nn.Linear(self.action_space_dimension, self.actions_count)

    def sample(self, mean: torch.Tensor, variance: torch.Tensor):
        '''
        Samples from the posterior distribution with given mean and variance

        :param mean: (..., action_space_dimension) tensor with posterior mean
        :param variance: (..., action_space_dimension) tensor with posterior variance
        :return: (..., action_space_dimension) tensor with points sampled from the posterior
        '''

        noise = torch.randn(mean.size(), dtype=torch.float32).cuda()
        sampled_points = noise * torch.sqrt(variance) + mean

        return sampled_points

    def split_batch(self, tensor: torch.Tensor):
        '''
        Splits a tensor in half following the first dimension
        Tensor must have an even number of elements in the first dimension
        :param tensor: (bs, ...) tensor to split
        :return: (bs/2, ...), (bs/2, ...) split tensors
        '''

        batch_size = tensor.size(0)
        assert(batch_size % 2 == 0)
        return tensor[:batch_size], tensor[batch_size:]

    def forward(self, states: torch.Tensor, states_attention: torch.Tensor) -> torch.Tensor:
        '''
        Computes actions corresponding to the state transition from predecessor to successor state

        :param states: (bs, observations_count, states_features, states_height, states_width) tensor
        :param states_attention: (bs, observations_count, 1, states_height, states_width) tensor with attention

        :return: action_probabilities, action_directions_distribution, sampled_action_directions,
                 action_states_distribution, sampled_action_states
                 (bs, observations_count - 1, actions_count) tensor with logits of probabilities for each action
                 (bs, observations_count - 1, 2, action_space_dimension) tensor posterior mean and variance for action directions
                 (bs, observations_count - 1, action_space_dimension) tensor with sampled action directions
                 (bs, observations_count, 2, action_space_dimension) tensor posterior mean and variance for action states
                 (bs, observations_count, action_space_dimension) tensor with sampled action states

        '''

        # Applies attention
        attentive_states = states * states_attention

        observations_count = attentive_states.size(1)
        
        flat_attentive_states = TensorFolder.flatten(attentive_states)

        x = self.residuals(flat_attentive_states)

        # Applies global average pooling
        x = self.gap(x)

        flat_states_mean = self.mean_fc(x.view(x.size(0), -1))
        flat_states_variance = torch.abs(self.variance_fc(x.view(x.size(0), -1))) # Maps the variance to positive values
        flat_states_distribution = torch.stack([flat_states_mean, flat_states_variance], dim=1)

        flat_sampled_states = self.sample(flat_states_mean, flat_states_variance)

        # Folds the tensors
        folded_states_mean = TensorFolder.fold(flat_states_mean, observations_count)
        folded_states_variance = TensorFolder.fold(flat_states_variance, observations_count)
        folded_states_distribution = TensorFolder.fold(flat_states_distribution, observations_count)
        folded_sampled_states = TensorFolder.fold(flat_sampled_states, observations_count)

        predecessor_mean, successor_mean = TensorSplitter.predecessor_successor_split(folded_states_mean)
        predecessor_variance, successor_variance = TensorSplitter.predecessor_successor_split(folded_states_variance)

        # The distribution of the difference vector is the difference of means and sum of variances
        action_directions_mean = successor_mean - predecessor_mean
        action_directions_variance = successor_variance + predecessor_variance
        action_directions_distribution = torch.stack([action_directions_mean, action_directions_variance], dim=2)
        sampled_action_directions = self.sample(action_directions_mean, action_directions_variance)

        flat_sampled_action_directions = TensorFolder.flatten(sampled_action_directions)
        # Computes the final action probabilities
        flat_action_probabilities = self.final_fc(flat_sampled_action_directions)
        folded_action_probabilities = TensorFolder.fold(flat_action_probabilities, observations_count - 1)

        return folded_action_probabilities, action_directions_distribution, sampled_action_directions, \
               folded_states_distribution, folded_sampled_states
























"""
    def bak_forward(self, predecessor_states: torch.Tensor, successor_states: torch.Tensor,
                predecessor_attention: torch.Tensor, successor_attention: torch.Tensor) -> torch.Tensor:
        '''
        Computes actions corresponding to the state transition from predecessor to successor state

        :param states: (bs, states_features, states_height, states_width) tensor
        :param successor_states: (bs, states_features, states_height, states_width) tensor
        :param states_attention: (bs, 1, states_height, states_width) tensor with attention
        :param successor_attention: (bs, 1, states_height, states_width) tensor with attention

        :return: action_probabilities, action_directions_distribution, sampled_action_directions,
                 predecessor_action_states_distribution, predecessor_sampled_action_states,
                 successor_action_states_distribution, successor_sampled_action_states,
                 (bs, actions_count) tensor with logits of probabilities for each action
                 (bs, 2, action_space_dimension) tensor posterior mean and variance for action directions
                 (bs, action_space_dimension) tensor with sampled action directions
                 (bs, 2, action_space_dimension) tensor posterior mean and variance for predecessor action states
                 (bs, action_space_dimension) tensor with sampled predecessor action states
                 (bs, 2, action_space_dimension) tensor posterior mean and variance for successor action states
                 (bs, action_space_dimension) tensor with sampled successor action states
        '''

        #TODO The strategy used for the propagation is not optimal because each state may be forwarded two times
        #in the successor and predecessor vectors which contain similar data

        # Stacks the inputs channelwise
        attentive_predecessor_states = predecessor_states * predecessor_attention
        attentive_successor_states = successor_states * successor_attention

        batch_size = attentive_predecessor_states.size(0)
        input_states = torch.cat([attentive_predecessor_states, attentive_successor_states], dim=0)

        x = self.residuals(input_states)

        # Applies global average pooling
        x = self.gap(x)

        states_mean = self.mean_fc(x.view(x.size(0), -1))
        states_variance = self.variance_fc(x.view(x.size(0), -1))
        states_distribution = torch.stack([states_mean, states_variance], dim=1)

        sampled_states = self.sample(states_mean, states_variance)

        # Splits the batch for predecessors and successors
        predecessor_states_distribution, successor_states_distribution = self.split_batch(states_distribution)
        predecessor_mean, successor_mean = self.split_batch(states_mean)
        predecessor_variance, successor_variance = self.split_batch(states_variance)
        predecessor_sampled_states, successor_sampled_states = self.split_batch(sampled_states)

        # The distribution of the difference vector is the difference of means and sum of variances
        action_directions_mean = successor_mean - predecessor_mean
        action_directions_variance = successor_variance + predecessor_variance
        action_directions_distribution = torch.stack([action_directions_mean, action_directions_variance], dim=1)
        sampled_action_directions = self.sample(action_directions_mean, action_directions_variance)

        # Computes the final action probabilities
        action_probabilities = self.final_fc(sampled_action_directions)

        return action_probabilities, action_directions_distribution, sampled_action_directions, \
               predecessor_states_distribution, predecessor_sampled_states, \
               successor_states_distribution, successor_sampled_states
"""