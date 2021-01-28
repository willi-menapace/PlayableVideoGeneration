from typing import Tuple, List

import torch
import torch.nn as nn

from model.layers.convolutional_lstm_cell import ConvLSTMCell


class ConvLSTM(nn.Module):
    '''
    A Convolutional LSTM
    '''

    def __init__(self, in_planes: int, out_planes: int, size: Tuple[int]):
        '''

        :param in_planes: Number of input channels
        :param out_planes: Number of output channels
        :param size: (height, width) of the input tensors
        '''
        super(ConvLSTM, self).__init__()

        #self.register_buffer('a', torch.zeros((1,)))
        # Initializes the memory cell
        self.cell = ConvLSTMCell(in_planes, out_planes)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.height = size[0]
        self.width = size[1]

        # Learnable initial cell states
        self.initial_hidden_state = nn.Parameter(torch.zeros(self.out_planes, self.height, self.width))
        self.initial_hidden_cell_state = nn.Parameter(torch.zeros(self.out_planes, self.height, self.width))

    def reinit_memory(self, batch_size: int):
        '''
        Initializes the cell state
        :param batch_size: Batch size of all the successive inputs until the next reinit_memory call
        :return:
        '''

        # Removes the stored state from the cell of present so that the next forward will reinitialize it
        # Warning: state is not created here directly, otherwise if the model is employed inside DataParallel,
        # the state would be created only in the original object rather than in the replicas on each GPU
        if hasattr(self, "current_hidden_state"):
            del self.current_hidden_state
            del self.current_hidden_cell_state

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Computes the successor states given the current inputs
        Current states are maintained implicitly and are reset through reinit_memory

        reinit_memory must have been called at least once before forward

        :param inputs: [(bs, features_i, height, width) \ (bs, features_i)] list of tensor which feature dimensions sum to in_planes

        :return: (bs, out_planes, height, width) tensor with the successor states
        '''

        batch_size = inputs[0].size(0)

        # Checks if state must be initialized
        # Initializes memory by repeating for each batch element the learned initial values
        if not hasattr(self, "current_hidden_state"):
            self.current_hidden_state = self.initial_hidden_state.repeat((batch_size, 1, 1, 1))
            self.current_hidden_cell_state = self.initial_hidden_cell_state.repeat((batch_size, 1, 1, 1))

        # Computes the next states and updates the memory
        cell_output = self.cell(inputs, self.current_hidden_state, self.current_hidden_cell_state)
        self.current_hidden_state, self.current_hidden_cell_state = cell_output

        return self.current_hidden_state
