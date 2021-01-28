from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from dataset.video import Video


class BatchElement:

    def __init__(self, observations: List[Tuple[Image.Image]], actions: List, rewards: List[int], dones: List[bool],
                 video: Video, initial_frame_index: int, transforms):
        '''
        Constructs a batch element

        :param observations: list of observations_count touples with observations_stacking frames each from the most recent to the oldest
        :param actions: list of observations_count actions
        :param rewards: list of observations_count rewards
        :param dones: list of observations_count booleans representing whether the episode has ended
        :param video: the original video object
        :param initial_frame_index: the index in the original video of the frame corresponding to the first observation
        :param transforms: transform to apply to each frame in the observations. Must return torch tensors
        '''

        self.observations_count = len(observations)
        self.observations_stacking = len(observations[0])

        if len(actions) != self.observations_count or len(actions) != self.observations_count or len(rewards) != self.observations_count or len(dones) != self.observations_count:
            raise Exception("Missing elements in the current batch")

        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.video = video
        self.initial_frame_index = initial_frame_index
        self.transforms = transforms

        self.observations = []
        for current_observation in observations:
            transformed_observation = [self.transforms(frame) for frame in current_observation]
            self.observations.append(transformed_observation)

class Batch:

    def __init__(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
                 videos: List[Video], initial_frames: List[int]):
        '''

        :param observations: (bs, observations_count, 3 * observations_stacking, h, w) tensor with observed images
        :param actions: (bs, observations_count) tensor with observed actions
        :param rewards: (bs, observations_count) tensor with observed rewards
        :param dones: (bs, observations_count) tensor with observed dones
        :param videos: list of original bs videos
        :param initial_frames: list of integers representing indexes in the original videos corresponding to the first frame
        '''

        self.size = actions.size(1)

        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.video = videos
        self.initial_frames = initial_frames

    def to_cuda(self):
        '''
        Transfers tensors to the gpu
        :return:
        '''
        self.observations = self.observations.cuda()
        self.actions = self.actions.cuda()
        self.rewards = self.rewards.cuda()
        self.dones = self.dones.cuda()

    def to_tuple(self, cuda=True) -> Tuple:
        '''
        Converts the batch to an input tuple
        :param cuda If True transfers the tensors to the gpu
        :return: (observations, actions, rewards, dones) tuple
        '''

        if cuda:
            self.to_cuda()

        return self.observations, self.actions, self.rewards, self.dones

    def pin_memory(self):
        self.observations.pin_memory()
        self.actions.pin_memory()
        self.rewards.pin_memory()
        self.dones.pin_memory()

        return self

def single_batch_elements_collate_fn(batch: List[BatchElement]) -> Batch:
    '''
    Creates a batch starting from single batch elements

    :param batch: List of batch elements
    :return: Batch representing the passed batch elements
    '''

    observations_tensor = torch.stack([torch.stack([torch.cat(current_stack) for current_stack in current_element.observations], dim=0) for current_element in batch], dim=0)
    actions_tensor = torch.stack([torch.tensor(current_element.actions, dtype=torch.int) for current_element in batch], dim=0)
    rewards_tensor = torch.stack([torch.tensor(current_element.rewards) for current_element in batch], dim=0)
    dones_tensor = torch.stack([torch.tensor(current_element.dones) for current_element in batch], dim=0)
    videos = [current_element.video for current_element in batch]
    initial_frames = [current_element.initial_frame_index for current_element in batch]

    return Batch(observations_tensor, actions_tensor, rewards_tensor, dones_tensor, videos, initial_frames)

def multiple_batch_elements_collate_fn(batch: List[Tuple[BatchElement]]) -> List[Batch]:
    '''
    Creates a batch starting from groups of corresponding batch elements

    :param batch: List of groups of batch elements
    :return: A List with cardinality equal to the number of batch elements of each group where
             the ith tuple item is the batch of all elements in the ith position in each group
    '''

    cardinality = len(batch[0])

    # Transforms the ith element of each group into its batch
    output_batches = []
    for idx in range(cardinality):
        # Extract ith element
        current_batch_elements = [current_elements_group[idx] for current_elements_group in batch]
        # Creates ith batch
        current_output_batch = single_batch_elements_collate_fn(current_batch_elements)
        output_batches.append(current_output_batch)

    return output_batches









