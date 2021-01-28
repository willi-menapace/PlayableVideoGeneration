from typing import Dict

import torch


class OneHotActionSampler:
    '''
    Module for sampling actions in one hot format
    '''

    def __init__(self):
        pass

    def __call__(self, log_probabilities: torch.Tensor, ground_truth: torch.Tensor):
        '''
        Returns the most probable action in form of a one hot vector

        :param log_probabilities: (bs, actions_count) tensor with logarithm of action probabilities
        :param ground_truth: (bs) tensor with ground truth action indexes
        :return: (bs, actions_count) tensor with one hot encodings of the most probable actions
        '''

        indexes = log_probabilities.argmax(dim=1)
        batch_size = log_probabilities.size(0)
        actions_count = log_probabilities.size(1)

        # Creates the one hot tensor
        onehot_tensor = torch.zeros((batch_size, actions_count), dtype=torch.float).cuda()

        # Populates it
        onehot_tensor.zero_()
        onehot_tensor.scatter_(1, indexes.reshape((-1, 1)).type(torch.LongTensor).cuda(), 1)

        return onehot_tensor


class GroundTruthActionSampler:
    '''
    Module for sampling ground truth actions
    '''

    def __init__(self, ground_truth_to_actions_mapping: Dict):
        '''

        :param ground_truth_to_actions_mapping: Dictionary that maps index of ground truth acctions to the
                                                corresponding indexes in the model action space
        '''
        self.mapping_dict = ground_truth_to_actions_mapping

    def translate_ground_truth_indexes(self, ground_truth: torch.Tensor) -> torch.Tensor:
        '''

        :param ground_truth: (bs) tensor with ground truth action indexes
        :return: (bs) tensor with action indexes in thr model action space
        '''

        output_tensor = ground_truth.clone()
        for gt_idx, idx in self.mapping_dict.items():
            output_tensor[ground_truth == gt_idx] = idx

        return output_tensor

    def __call__(self, log_probabilities: torch.Tensor, ground_truth: torch.Tensor):
        '''
        Returns the most probable action in form of a one hot vector

        :param log_probabilities: (bs, actions_count) tensor with logarithm of action probabilities
        :param ground_truth: (bs) tensor with ground truth action indexes
        :return: (bs, actions_count) tensor with one hot encodings of the most probable actions
        '''

        batch_size = log_probabilities.size(0)
        actions_count = log_probabilities.size(1)

        translated_ground_truth = self.translate_ground_truth_indexes(ground_truth)

        # Creates the one hot tensor
        onehot_tensor = torch.zeros((batch_size, actions_count), dtype=torch.float).cuda()

        # Populates it
        onehot_tensor.zero_()
        onehot_tensor.scatter_(1, translated_ground_truth.reshape((-1, 1)).type(torch.LongTensor).cuda(), 1)

        return onehot_tensor
