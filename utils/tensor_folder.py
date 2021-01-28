from typing import List

import torch
import torch.nn as nn

class TensorFolder:

    @staticmethod
    def flatten(tensor: torch.Tensor) -> torch.Tensor:
        '''
        Flattens the first two dimensions of the tensor

        :param tensor: (dim1, dim2, ...) tensor
        :return: (dim1 * dim2, ...) tensor
        '''

        tensor_size = list(tensor.size())
        flattened_tensor = torch.reshape(tensor, tuple([-1] + tensor_size[2:]))

        return flattened_tensor

    @staticmethod
    def flatten_list(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        '''
        Applies flatten to all elements in the sequence
        See flatten for additional details
        '''

        flattened_tensors = [TensorFolder.flatten(current_tensor) for current_tensor in tensors]
        return flattened_tensors

    @staticmethod
    def fold(tensor: torch.Tensor, second_dimension_size: torch.Tensor) -> torch.Tensor:
        '''
        Separates the first tensor dimension into two separate dimensions of the given size

        :param tensor: (dim1 * second_dimension_size, ...) tensor
        :param second_dimension_size: the wished second dimension size for the output tensor
        :return: (dim1, second_dimension_size, ...) tensor
        '''

        tensor_size = list(tensor.size())
        first_dimension_size = tensor_size[0]

        # Checks sizes
        if first_dimension_size % second_dimension_size != 0:
            raise Exception(f"First dimension {first_dimension_size} is not a multiple of {second_dimension_size}")

        folded_first_dimension_size = first_dimension_size // second_dimension_size
        tensor = torch.reshape(tensor, ([folded_first_dimension_size, second_dimension_size] + tensor_size[1:]))
        return tensor

    @staticmethod
    def fold_list(tensors: List[torch.Tensor], second_dimension_size: torch.Tensor) -> List[torch.Tensor]:
        '''
        Applies fold to each element in the sequence
        See fold for additional details
        '''

        folded_tensors = [TensorFolder.fold(current_tensor, second_dimension_size) for current_tensor in tensors]
        return folded_tensors
