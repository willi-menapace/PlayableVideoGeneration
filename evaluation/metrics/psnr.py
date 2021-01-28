import torch
import torch.nn as nn
import numpy as np

class PSNR(nn.Module):

    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor, range=1.0) -> torch.Tensor:
        '''
        Computes the psnr between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :param range: The maximum value used to represent each pixel
        :return: (bs, observations_count) tensor with psnr for each observation
        '''

        # Constant for numerical stability
        EPS = 1e-8

        reference_observations = reference_observations / range
        generated_observations = generated_observations / range

        mse = torch.mean((reference_observations - generated_observations) ** 2, dim=[2, 3, 4])
        score = - 10 * torch.log10(mse + EPS)

        return score