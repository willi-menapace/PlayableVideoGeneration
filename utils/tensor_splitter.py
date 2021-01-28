import torch
import torch.nn as nn

class TensorSplitter:

    @staticmethod
    def predecessor_successor_split(tensor: torch.Tensor) -> torch.Tensor:
        '''
        Splits a tensor into the second dimension predecessors and successors

        :param tensor: (dim1, dim2, ...) tensor
        :return: (dim1, 0:dim2-1, ...), (dim1, 1:dim2, ...) tensor
        '''

        predecessor_tensor = tensor[:, :-1]
        successor_tensor = tensor[:, 1:]

        return predecessor_tensor, successor_tensor


