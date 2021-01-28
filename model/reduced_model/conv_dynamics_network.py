from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.convolutional_lstm import ConvLSTM
from model.layers.convolutional_lstm_cell import ConvLSTMCell
from model.layers.residual_block import ResidualBlock
from model.layers.same_block import SameBlock
from model.layers.up_block import UpBlock


class ConvDynamicsNetwork(nn.Module):
    '''
    Model that predicts the future state given the current state and an action
    '''

    def __init__(self, config):
        super(ConvDynamicsNetwork, self).__init__()

        self.hidden_state_size = config["model"]["dynamics_network"]["hidden_state_size"]
        self.random_noise_size = config["model"]["dynamics_network"]["random_noise_size"]
        self.state_resolution = config["model"]["representation_network"]["state_resolution"]
        self.state_features = config["model"]["representation_network"]["state_features"]

        actions_count = config["data"]["actions_count"]
        actions_space_dimension = config["model"]["action_network"]["action_space_dimension"]

        # Dimension of the actions embeddings given along with the input. Noise is not given
        auxiliary_input_size = actions_count + actions_space_dimension

        # The recurrent layers used by the model
        self.recurrent_layers = [
            ConvLSTM(self.state_features + auxiliary_input_size, self.hidden_state_size, self.state_resolution),
            ConvLSTM(2 * self.hidden_state_size + auxiliary_input_size, 2 * self.hidden_state_size, (self.state_resolution[0] // 2, self.state_resolution[1] // 2)),
            ConvLSTM(self.hidden_state_size + auxiliary_input_size, self.hidden_state_size, self.state_resolution)
        ]

        # Blocks with the recurrent layers and their normalization + activation
        self.recurrent_layers_blocks = nn.ModuleList([
                nn.Sequential(self.recurrent_layers[0], nn.BatchNorm2d(self.hidden_state_size)),
                nn.Sequential(self.recurrent_layers[1], nn.BatchNorm2d(2 * self.hidden_state_size)),
                nn.Sequential(self.recurrent_layers[2], nn.BatchNorm2d(self.hidden_state_size))
            ])

        self.non_recurrent_blocks = nn.ModuleList([
            SameBlock(self.hidden_state_size + auxiliary_input_size, 2 * self.hidden_state_size, downsample_factor=2),
            UpBlock(2 * self.hidden_state_size + auxiliary_input_size, self.hidden_state_size, upscaling_mode="bilinear", late_upscaling=True),
            SameBlock(self.hidden_state_size + auxiliary_input_size, self.hidden_state_size, downsample_factor=1),
        ])

    def reinit_memory(self, batch_size: int):
        '''
        Initializes the state of the recurrent cells
        :param batch_size: Batch size of all the successive inputs until the next reinit_memory call
        :return:
        '''

        # Initializes memory
        for current_layer in self.recurrent_layers:
            current_layer.reinit_memory(batch_size)

    def make_2d_tensor(self, tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
        '''
        Transforms a 1d tensor into a 2d tensor of specified dimensions

        :param tensor: (bs, features) tensor
        :return: (bs, features, height, width) tensor with repeated features along the spatial dimensions
        '''

        tensor = tensor.unsqueeze(dim=-1).unsqueeze(dim=-1)  # Adds two final dimensions for broadcast
        tensor = tensor.repeat((1, 1, height, width))  # Repeats along the spatial dimensions

        return tensor

    def channelwise_concat(self, inputs: List[torch.Tensor]):
        '''
        Concatenates all inputs tensors channelwise

        :param inputs: [(bs, features_i, height, width) \ (bs, features_i)] list of tensor which feature dimensions sum to in_planes
        :return:
        '''

        # Infers the target spatial dimensions
        height = 0
        width = 0
        for current_tensor in inputs:
            if len(current_tensor.size()) == 4:
                height = current_tensor.size(2)
                width = current_tensor.size(3)
                break
        if height == 0 or width == 0:
            raise Exception("No tensor in the inputs has a spatial dimension. Ensure at least one tensor represents a tensor with spatial dimensions")

        # Expands tensors to spatial dimensions
        expanded_tensors = []
        for current_tensor in inputs:
            if len(current_tensor.size()) == 4:
                expanded_tensors.append(current_tensor)
            elif len(current_tensor.size()) == 2:
                expanded_tensors.append(self.make_2d_tensor(current_tensor, height, width))
            else:
                raise Exception("Expected tensors with 2 or 4 dimensions")

        # Concatenates tensors channelwise
        concatenated_tensor = torch.cat(expanded_tensors, dim=1)

        return concatenated_tensor

    def forward(self, states: torch.Tensor, actions: torch.Tensor, variations: torch.Tensor, random_noise: torch.Tensor) -> torch.Tensor:
        '''
        Computes the successor states given the selected actions and noise
        Current states are maintained implicitly and are reset through reinit_memory
        reinit_memory must have been called at least once before forward

        :param states: (bs, states_features, states_height, states_width) tensor
        :param actions: (bs, actions_count) tensor with actions probabilities
        :param variations: (bs, action_space_dimension) tensor with action variations
        :param random_noise: (bs, random_noise_size, states_height, states_width) tensor with random noise

        :return: (bs, hidden_state_size) tensor with the successor states
        '''

        # Passes the input tensors through each block, concatenating the auxiliary inputs at each step
        states = self.recurrent_layers_blocks[0]([states, actions, variations])                          # size / 1
        states = self.non_recurrent_blocks[0](self.channelwise_concat([states, actions, variations]))    # size / 1
        states = self.recurrent_layers_blocks[1]([states, actions, variations])                          # size / 2
        states = self.non_recurrent_blocks[1](self.channelwise_concat([states, actions, variations]))    # size / 2
        states = self.recurrent_layers_blocks[2]([states, actions, variations])                          # size / 1
        states = self.non_recurrent_blocks[2](self.channelwise_concat([states, actions, variations]))    # size / 1

        return states