from typing import Dict

import torch


class ZeroActionVariationSampler:
    '''
    Module for sampling actions in one hot format
    '''

    def __init__(self):
        pass

    def __call__(self, sampled_action_directions: torch.Tensor, action_samples: torch.Tensor):
        '''
        Returns the most probable action in form of a one hot vector

        :param sampled_action_directions: (bs, space_dimensions) tensor with each point
        :param action_samples: (bs, actions_count) tensor with cluster assignment probabilities in 0, 1 for each point
        :return: (bs, space_dimensions) tensor sampled_action_variation

        '''

        # Zeroes the variations
        return sampled_action_directions * 0
