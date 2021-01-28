from typing import List, Dict, Tuple

import numpy as np
import glob
from PIL import Image
import os
import pickle

class Video:
    '''
     A video from the dataset
     Metadata are always kept into memory, while frames are loaded from disk on demand only
    '''

    actions_filename = "actions.pkl"
    rewards_filename = "rewards.pkl"
    metadata_filename = "metadata.pkl"
    dones_filename = "dones.pkl"

    def __init__(self):
        self.frames = None
        self.actions = None
        self.rewards = None
        self.metadata = None

    def add_content(self, frames: List[Image.Image], actions: List[int], rewards: List[float], metadata: List[Dict], dones: List[bool]):
        '''
        Adds the contents to the video
        :param frames: list of all video frames
        :param actions: list of the actions selected in the current frame
        :param rewards: list of the reward generated for arriving in the current frame
        :param metadata: list of metadata associated to the current frame
        :param dones: list of booleans representing whether the current observation was the last of the episode
        :return:
        '''
        if len(frames) != len(actions) or len(frames) != len(rewards) or len(frames) != len(metadata) or len(frames) != len(dones):
            raise Exception("All arguments must have the same length")

        self.frames = frames
        self.actions = actions
        self.rewards = rewards
        self.metadata = metadata
        self.dones = dones

        # Sets default values in the metadata if needed
        self.check_metadata_and_set_defaults()

        self.extension = None
        self.frames_path = None

    def _index_to_filename(self, idx):
        return f'{idx:05}'

    def check_none_coherency(self, sequence):
        '''
        Checks that the sequence either has all values set to None or to a not None value
        Raises an exception if the sequence does not satisfy the criteria
        :param sequence: the sequence to check
        :return:
        '''

        has_none = False
        has_not_none = False

        for element in sequence:
            if element is None:
                has_none = True
            else:
                has_not_none = True
            if has_none and has_not_none:
                raise Exception(f"Video dataset at {self.frames_path} metadata error: both None and not None data are present")

    def check_metadata_and_set_defaults(self):
        '''
        Checks the medatata and sets default values if None are present
        :return:
        '''

        # Checks coherency of None values in the metadata
        self.check_none_coherency(self.actions)
        self.check_none_coherency(self.rewards)
        self.check_none_coherency(self.metadata)
        self.check_none_coherency(self.dones)

        if self.actions[0] is None:
            self.actions = [0] * len(self.actions)
        if self.rewards[0] is None:
            self.rewards = [0.0] * len(self.rewards)
        if self.metadata[0] is None:
            self.metadata = [{}] * len(self.metadata)
        if self.dones[0] is None:
            self.dones = [False] * len(self.dones)


    def load(self, path):

        if not os.path.isdir(path):
            raise Exception(f"Cannot load video: '{path}' is not a directory")

        # Frames are not load into memory
        self.frames_path = path

        # Load data as pickle objects
        with open(os.path.join(path, Video.actions_filename), 'rb') as f:
            self.actions = pickle.load(f)
        with open(os.path.join(path, Video.rewards_filename), 'rb') as f:
            self.rewards = pickle.load(f)
        with open(os.path.join(path, Video.metadata_filename), 'rb') as f:
            self.metadata = pickle.load(f)
        with open(os.path.join(path, Video.dones_filename), 'rb') as f:
            self.dones = pickle.load(f)

        frames_count = len(self.actions)
        if frames_count != len(self.rewards) or frames_count != len(self.metadata) or frames_count != len(self.dones):
            raise Exception("Read data have inconsistent number of frames")

        # Sets detault values in the metadata if needed
        self.check_metadata_and_set_defaults()

        # Discover extension of videos
        padded_index = self._index_to_filename(0)
        results = glob.glob(os.path.join(path, f'{padded_index}.*'))
        if len(results) != 1:
            raise Exception("Could not find first video frame")
        extension = results[0].split(".")[-1]
        self.extension = extension


    def get_frames_count(self) -> int:
        if self.actions is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")

        return len(self.actions)


    def get_frame_at(self, idx: int) -> Image:
        '''
        Returns the frame corresponding to the specified index

        :param idx: index of the frame to retrieve in [0, frames_count-1]
        :return: The frame at the specified index
        '''
        if self.actions is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")
        if idx < 0 or idx >= len(self.actions):
            raise Exception(f"Index {idx} is out of range")

        # If frames are load into memory
        if self.frames != None:
            return self.frames[idx]

        padded_index = self._index_to_filename(idx)
        filename = os.path.join(self.frames_path, f'{padded_index}.{self.extension}')
        image = Image.open(filename)
        image = self.remove_transparency(image)
        return image

    def remove_transparency(self, image, bg_colour=(255, 255, 255)):

        # Only process if image has transparency (http://stackoverflow.com/a/1963146)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):

            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = image.convert('RGBA').split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            bg = Image.new("RGBA", image.size, bg_colour + (255,))
            bg.paste(image, mask=alpha)
            bg = bg.convert("RGB")
            return bg
        else:
            return image

    def subsample_split_resize(self, frame_skip: int, output_sequence_length: int, crop_size: Tuple[int], target_size: Tuple[int]) -> List:
        '''
        Splits the current sequence into a number of sequences of the specified length, skipping the specified number
        of frames in the source sequence between successive frames in the target sequence
        Resizes the output sequence to the target_size

        :param frame_skip: frames to skip in the source sequence between successive frames in the target sequence
        :param output_sequence_length: number of frames in each output sequence. -1 if length should not be modified
        :param crop_size: (left_index, upper_index, right_index, lower_index) size of the crop to take before resizing
        :param target_size: (width, height) size of the frames in the output sequence
        :return: List of videos representing the split and subsampled source video
        '''

        # Subsamples the video
        all_frames = [self.get_frame_at(idx) for idx in range(0, self.get_frames_count(), frame_skip + 1)]
        all_actions = self.actions[::frame_skip + 1]
        all_rewards = self.rewards[::frame_skip + 1]
        all_metadata = self.metadata[::frame_skip + 1]
        all_dones = self.dones[::frame_skip + 1]

        # Crops the frames
        all_frames = [frame.crop(crop_size) for frame in all_frames]

        # Resizes the video if needed
        original_width, original_height = all_frames[0].size
        if original_width != target_size[0] or original_height != target_size[1]:
            all_frames = [frame.resize(target_size, Image.BICUBIC) for frame in all_frames]

        split_videos = []

        # Applies length splitting if needed
        if output_sequence_length > 0:
            # Splits the subsampled video in constant length sequences
            total_frames = len(all_frames)
            for current_idx in range(0, total_frames, output_sequence_length):
                if current_idx + output_sequence_length < total_frames:
                    current_frames = all_frames[current_idx:current_idx + output_sequence_length]
                    current_actions = all_actions[current_idx:current_idx + output_sequence_length]
                    current_rewards = all_rewards[current_idx:current_idx + output_sequence_length]
                    current_metadata = all_metadata[current_idx:current_idx + output_sequence_length]
                    current_dones = all_dones[current_idx:current_idx + output_sequence_length]

                    current_video = Video()
                    current_video.add_content(current_frames, current_actions, current_rewards, current_metadata, current_dones)
                    split_videos.append(current_video)

        # Otherwise return the video in original length
        else:
            current_video = Video()
            current_video.add_content(all_frames, all_actions, all_rewards, all_metadata, all_dones)
            split_videos.append(current_video)

        return split_videos

    def save_moco(self, path: str, extension="png", target_size=None):
        '''
        Saves a video to the moco format. The video must already be present on disk
        :param path: The location where to save the video in moco format
        :param extension: The extension to use for image files
        :param target_size: (witdh, height) size for the dataset
        :return:
        '''

        if os.path.exists(path):
            raise Exception(f"A directory at '{path}' already exists")

        all_frames = [self.get_frame_at(idx) for idx in range(self.get_frames_count())]

        # Resizes the images if needed
        if target_size is not None:
            all_frames = [current_frame.resize(target_size) for current_frame in all_frames]

        widths, heights = zip(*(i.size for i in all_frames))

        total_width = sum(widths)
        max_height = max(heights)

        concatenated_frame = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for current_image in all_frames:
            concatenated_frame.paste(current_image, (x_offset, 0))
            x_offset += current_image.size[0]

        concatenated_frame.save(f"{path}.{extension}")

    def save(self, path: str, extension="png"):
        if self.actions is None:
            raise Exception("Video has not been initialized. Did you forget to call add_content()?")
        if os.path.isdir(path):
            raise Exception(f"A directory at '{path}' already exists")

        # Creates the directory
        os.mkdir(path)

        # Save all frames
        for idx, frame in enumerate(self.frames):
            padded_index = self._index_to_filename(idx)
            filename = os.path.join(path, f'{padded_index}.{extension}')
            frame.save(filename)

        # Saves other data as pickle objects
        with open(os.path.join(path, Video.actions_filename), 'wb') as f:
            pickle.dump(self.actions, f)
        with open(os.path.join(path, Video.rewards_filename), 'wb') as f:
            pickle.dump(self.rewards, f)
        with open(os.path.join(path, Video.metadata_filename), 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(os.path.join(path, Video.dones_filename), 'wb') as f:
            pickle.dump(self.dones, f)

