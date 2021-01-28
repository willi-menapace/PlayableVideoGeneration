import os
from typing import Set, List, Dict

import torch
import torchvision.transforms as tf
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.batching import BatchElement, single_batch_elements_collate_fn, multiple_batch_elements_collate_fn
from dataset.transforms import TransformsGenerator
from dataset.video import Video


class VideoDataset(Dataset):
    '''
    Dataset of video objects
    Expects a directory where each children directory represents a Video object on disk
    '''

    def __init__(self, path, batching_config: Dict, final_transform, allowed_videos=None):
        '''
        Initializes the dataset with the videos in a directory

        :param path: path to the root of the dataset
        :param batching_config: Dict with the batching parameters to use for sampling
        :param final_transform: transformation to apply to each frame
        :param allowed_videos: set of video names allowed to be part of the dataset.
                               if not None only videos in this set are included in the dataset
        '''

        if not os.path.isdir(path):
            raise Exception(f"Dataset directory '{path}' is not a directory")

        self.batching_config = batching_config

        # number of frames that compose each observation
        self.observations_stacking = batching_config['observation_stacking']
        # how many frames to skip between each observation
        self.skip_frames = batching_config['skip_frames']
        self.final_transform = final_transform

        # Reads the videos in the root
        self.all_videos = self.read_all_videos(path, allowed_videos)

        self.observations_count = None
        # number of observations to include in each dataset sample
        self.set_observations_count(batching_config['observations_count'])

    def set_observations_count(self, observations_count: int):
        '''
        Changes the number of observations in each future returned sequence

        :param observations_count: Number of observations in each future returned sequence
        :return:
        '''

        # Perform changes only if the parameter differs
        if self.observations_count is None or self.observations_count != observations_count:
            self.observations_count = observations_count

            self.available_samples_list = self.compute_available_samples_per_video()
            self.total_available_samples = sum(self.available_samples_list)


    def read_all_videos(self, path: str, allowed_videos: Set[str]) -> List[Video]:
        '''
        Reads all the allowed videos in the specified path

        :param path: path where videos are stored
        :param allowed_videos: set of video names allowed to be part of the dataset
                               if None all videos are included in the dataset
        :return:
        '''

        all_videos = []
        contents = sorted(list(os.listdir(path)))

        # Allow everything if no limitations are specified
        if allowed_videos is None:
            allowed_videos = contents

        for current_file in contents:
            current_file_path = os.path.join(path, current_file)
            print(f"- Loading video at '{current_file_path}'")
            if os.path.isdir(current_file_path) and current_file in allowed_videos:
                current_video = Video()
                current_video.load(current_file_path)
                all_videos.append(current_video)

        return all_videos

    def compute_available_samples_per_video(self) -> List[int]:
        '''
        Computes how many samples can be drawn from the video sequences

        :return: list with an integer for each video representing how many samples can be drawn
        '''

        available_samples = []

        # Number of frames in the original video each sample will span
        sample_block_size = self.observations_count + (self.observations_count - 1) * self.skip_frames

        for current_video in self.all_videos:
            frames_count = current_video.get_frames_count()
            current_samples = frames_count - sample_block_size + 1
            available_samples.append(current_samples)

        return available_samples

    def __len__(self):
        return self.total_available_samples

    def __getitem__(self, index):

        if index >= self.total_available_samples:
            raise Exception(f"Requested sample at index {index} is out of range")

        video_index = 0
        video_initial_frame = 0

        # Searches the video and the frame index in that video where to start extracting the sequence
        passed_samples = 0
        for search_index, current_available_samples in enumerate(self.available_samples_list):
            if passed_samples + current_available_samples > index:
                video_index = search_index
                video_initial_frame = index - passed_samples
                break
            passed_samples += current_available_samples

        current_video = self.all_videos[video_index]
        observation_indexes = []
        for i in range(self.observations_count):
            observation_indexes.append(video_initial_frame + i * (self.skip_frames + 1))


        min_frame = video_initial_frame % (self.skip_frames + 1) # The minimum frame for which the preceding would not be part of the video
        all_frames_indexes = [[max(current_observation_index - i * (self.skip_frames + 1), min_frame) for i in range(self.observations_stacking)] for current_observation_index in observation_indexes]

        all_frames = [[current_video.get_frame_at(index) for index in current_observation_stack] for current_observation_stack in all_frames_indexes]
        # Action is the one selected in the frame where the observation took place
        all_actions = [current_video.actions[current_index] for current_index in observation_indexes]
        # The reward is the reward obtained for arriving in the current observation frame from the previous summing also the reward from the frames that were skipped
        all_rewards = [sum(current_video.rewards[max(current_index - self.skip_frames, 0):current_index + 1]) for current_index in observation_indexes]
        all_dones = [current_video.dones[current_index] for current_index in observation_indexes]

        plain_batch_element = BatchElement(all_frames, all_actions, all_rewards, all_dones, current_video, video_initial_frame, self.final_transform)

        return plain_batch_element


if __name__ == "__main__":

    import torch
    import torchvision
    import torchvision.transforms as transforms

    dataset_path = 'data/tennis_v4/fixed_length_subsampled_test'
    batching_config = {
        'observations_count': 16,
        'observation_stacking': 4,
        'skip_frames': 0,
    }


    transformations = transforms.ToTensor()

    dataset = VideoDataset(dataset_path, batching_config, transformations)

    print(f'- Dataset length: {len(dataset)}')
    #for i in range(len(dataset)):
    #    print(f"Extracting element [{i}]")
    #    current_element = dataset[i]

    print("- Creating dataloader")
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=single_batch_elements_collate_fn, num_workers=1, pin_memory=True)


    for idx, batch in enumerate(dataloader):
        print(f'Batch succesfully extracted [{idx}/{len(dataloader)}]')
        break

    dataset.set_observations_count(6)

    for idx, batch in enumerate(dataloader):
        print(f'Batch succesfully extracted [{idx}/{len(dataloader)}]')




