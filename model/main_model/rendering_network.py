from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul

from model.layers.final_block import FinalBlock
from model.layers.residual_block import ResidualBlock
from model.layers.up_block import UpBlock


class RenderingNetwork(nn.Module):
    '''
    Model that reconstructs the frame associated to a hidden state
    '''

    def __init__(self, config):
        super(RenderingNetwork, self).__init__()

        self.config = config
        # Shape of the input flattened tensor
        self.hidden_state_size = config["model"]["dynamics_network"]["hidden_state_size"]

        bottleneck_block_list = [

        ]

        upsample_block_list = [
            nn.Sequential(UpBlock(128, 128, scale_factor=2, upscaling_mode="bilinear"),  # res * 4
                          ResidualBlock(128, 128, downsample_factor=1)),
            nn.Sequential(UpBlock(128, 64, scale_factor=2, upscaling_mode="bilinear"),  # res * 8
                          ResidualBlock(64, 64, downsample_factor=1)),
            UpBlock(64, 32, scale_factor=2, upscaling_mode="bilinear"),  # res * 16
        ]

        final_block_list = [
            FinalBlock(128, 3, kernel_size=3, padding=1),
            FinalBlock(64, 3, kernel_size=3, padding=1),
            FinalBlock(32, 3, kernel_size=7, padding=3)
        ]

        self.bottleneck_blocks = nn.Sequential(*bottleneck_block_list)
        self.upsample_blocks = nn.ModuleList(upsample_block_list)
        self.final_blocks = nn.ModuleList(final_block_list)

        if len(upsample_block_list) != len(final_block_list):
            raise Exception("Rendering network specifies a number of upsampling blocks that differs from the number of final blocks")


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        '''
        Computes the frames corresponding to each state at multiple resolutions

        :param hidden_states: (bs, hidden_state_size, state) tensor
        :return: (bs, 3, height, width), [(bs, 3, height/2^i, width/2^i) for i in range(num_upsample_blocks)]
        '''

        current_features = self.bottleneck_blocks(hidden_states)

        reconstructed_observations = []
        for upsample_block, final_block in zip(self.upsample_blocks, self.final_blocks):
            # Upsample the features
            current_features = upsample_block(current_features)
            # Transform them in the corresponding resolution image
            current_reconstructed_observation = final_block(current_features)
            reconstructed_observations.append(current_reconstructed_observation)

        reconstructed_observations = list(reversed(reconstructed_observations)) # Inverts from high res to low res
        return reconstructed_observations[0], reconstructed_observations
