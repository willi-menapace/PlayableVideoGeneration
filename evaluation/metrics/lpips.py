import torch
import torch.nn as nn
import numpy as np
import lpips

from utils.tensor_folder import TensorFolder


class LPIPS(nn.Module):

    def __init__(self):
        super(LPIPS, self).__init__()

        self.metric = lpips.LPIPS(net='vgg').cuda()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes the psnr between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with psnr for each observation
        '''


        observations_count = reference_observations.size(1)

        all_lpips = []
        for observation_idx in range(observations_count):
            current_reference_observations = reference_observations[:, observation_idx]
            current_generated_observations = generated_observations[:, observation_idx]

            lpips = self.metric(current_reference_observations, current_generated_observations, normalize=True)
            lpips = lpips.reshape((-1)) # Squeezes all the dimensions since only the first is != 1
            all_lpips.append(lpips)

        return torch.stack(all_lpips, axis=1)