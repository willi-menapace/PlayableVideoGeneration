import torch
import torch.nn as nn
import numpy as np
from piq import ssim

from utils.tensor_folder import TensorFolder


class SSIM(nn.Module):

    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor, range=1.0) -> torch.Tensor:
        '''
        Computes the ssim between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :param range: The maximum value used to represent each pixel
        :return: (bs, observations_count) tensor with ssim for each observation
        '''

        # Flattens observations and then folds the results
        observations_count = reference_observations.size(1)
        flattened_reference_observations = TensorFolder.flatten(reference_observations)
        flattened_generated_observations = TensorFolder.flatten(generated_observations)
        flattened_ssim = ssim(flattened_generated_observations, flattened_reference_observations, range, reduction="none")
        folded_ssim = TensorFolder.fold(flattened_ssim, observations_count)

        return folded_ssim


