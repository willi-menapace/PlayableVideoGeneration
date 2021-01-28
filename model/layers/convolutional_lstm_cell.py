from typing import List

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    '''
    A Convolutional LSTM Cell
    '''

    def __init__(self, in_planes: int, out_planes: int):
        '''

        :param in_planes: Number of input channels
        :param out_planes: Number of output channels
        '''
        super(ConvLSTMCell, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.input_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.forget_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.cell_gate = nn.Conv2d(in_planes + self.out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

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
        total_features = concatenated_tensor.size(1)
        if total_features != self.in_planes + self.out_planes:
            raise Exception(f"The input tensors features sum to {total_features}, but layer takes {self.in_planes} features as input")

        return concatenated_tensor

    def forward(self, inputs: List[torch.Tensor], hidden_states: torch.Tensor, hidden_cell_states: torch.Tensor) -> torch.Tensor:
        '''
        Computes the successor states given the inputs

        :param inputs: [(bs, features_i, height, width) \ (bs, features_i)] list of tensor which feature dimensions sum to in_planes
        :param hidden_states: (bs, out_planes, height, width) tensor with hidden state
        :param hidden_cell_states: (bs, out_planes, height, width) tensor with hidden cell state

        :return: (bs, out_planes, height, width), (bs, out_planes, height, width) tensors with hidden_state and hidden_cell_state
        '''

        inputs.append(hidden_states)  # Also hidden states must be convolved with the input
        concatenated_input = self.channelwise_concat(inputs)

        # Processes the gates
        i = torch.sigmoid(self.input_gate(concatenated_input))
        f = torch.sigmoid(self.forget_gate(concatenated_input))
        o = torch.sigmoid(self.output_gate(concatenated_input))
        c = torch.tanh(self.cell_gate(concatenated_input))

        # Computes successor states
        successor_hidden_cell_states = f * hidden_cell_states + \
                                       i * c

        successor_hidden_state = o * torch.tanh(successor_hidden_cell_states)

        return successor_hidden_state, successor_hidden_cell_states