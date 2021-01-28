import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image

from evaluation.metrics.motion_mask import MotionMaskCalculator
from utils.tensor_displayer import TensorDisplayer


class MotionMaskedMSE(nn.Module):

    def __init__(self):
        super(MotionMaskedMSE, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes the mean squared error between the reference and the generated observations

        :param reference_observations: (bs, observations_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with MSE for each observation
        '''

        motion_mask = MotionMaskCalculator.compute_frame_difference_motion_mask(reference_observations)
        differences = (reference_observations - generated_observations).pow(2) * motion_mask

        return torch.mean(differences, dim=[2, 3, 4])
