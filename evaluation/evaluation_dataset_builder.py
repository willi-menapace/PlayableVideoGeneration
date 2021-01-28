import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn
from dataset.video import Video
from dataset.video_dataset import VideoDataset
from evaluation.action_sampler import OneHotActionSampler
from evaluation.action_variation_sampler import ZeroActionVariationSampler


class EvaluationDatasetBuilder:
    '''
    Helper class for model evaluation
    '''

    def __init__(self, config, dataset: VideoDataset, logger, logger_prefix="test"):

        self.config = config
        self.logger = logger
        self.logger_prefix = logger_prefix
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.config["evaluation"]["batching"]["batch_size"], shuffle=False, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)

        self.output_path = self.config["logging"]["evaluation_dataset_directory"]
        self.ground_truth_observations_init = self.config["evaluation_dataset"]["ground_truth_observations_init"]

        self.action_variation_sampler = ZeroActionVariationSampler()
        self.temperature = self.config["training"]["gumbel_temperature_end"]

    def build(self, model):
        '''
        Builds the evaluation dataset

        :param model: The model to use for building the evaluation dataset
        :return:
        '''

        all_videos = []

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):

                # Performs inference using the sampler for action variations
                batch_tuple = batch.to_tuple()
                results = model(batch_tuple, ground_truth_observations_init=self.ground_truth_observations_init,
                                action_sampler=OneHotActionSampler(), action_variation_sampler=self.action_variation_sampler,
                                gumbel_temperature=self.temperature)

                # Extracts the results
                reconstructed_observations, multiresolution_reconstructed_observations, reconstructed_states, states, hidden_states, selected_actions, action_logits, action_samples_distribution, \
                attention, reconstructed_attention, action_directions_distribution, sampled_action_directions, \
                action_states_distribution, sampled_action_states, action_variations, \
                reconstructed_action_logits, \
                reconstructed_action_directions_distribution, reconstructed_sampled_action_directions, \
                reconstructed_action_states_distribution, reconstructed_sampled_action_states, \
                *other_results = results

                # Pads the reconstructed observations with the first ground truth image
                reconstructed_observations = torch.cat([batch_tuple[0][:, 0:1, 0:3], reconstructed_observations], dim=1)
                # Normalizes the range of the observations
                reconstructed_observations = self.check_and_normalize_range(reconstructed_observations)

                # Converts to numpy
                reconstructed_observations = reconstructed_observations.cpu().numpy()
                reconstructed_observations = np.moveaxis(reconstructed_observations, 2, -1)  # Moves the channels as the last dimension
                selected_actions = selected_actions.cpu().numpy()                      # The actions that have been selected
                sampled_action_directions = sampled_action_directions.cpu().numpy()    # The action vector

                # Builds the video objects for the current batch
                current_videos = self.predictions_to_videos(reconstructed_observations, selected_actions, sampled_action_directions)
                all_videos.extend(current_videos)

        # Creates the dataset
        self.create_dataset(self.output_path, all_videos)

    def predictions_to_videos(self, images: np.ndarray, actions: np.ndarray, encoded_mus: np.ndarray) -> List[Video]:
        '''

        :param images: (bs, observations_count, height, width, channels) tensor
        :param actions: (bs, observations_count - 1) tensor
        :param encoded_mus: (bs, observations_count, mus_dimensions) tensor
        :return:
        '''

        images = (images * 255).astype(np.uint8)

        batch_size, sequence_length, height, width, channels = images.shape
        if actions.shape[0] != batch_size:
            raise Exception(f"Images have batch size {batch_size} but actions have batch size {actions.shape[0]}")
        if actions.shape[1] != sequence_length - 1:
            raise Exception(
                f"Images have sequence length {sequence_length} but actions have sequence length {actions.shape[1]}")

        all_videos = []
        # Transforms a sequence at a time into a video
        for sequence_idx in range(batch_size):
            current_images = images[sequence_idx]
            current_actions = actions[sequence_idx].tolist()
            current_encoded_mus = encoded_mus[sequence_idx].tolist()
            current_images = [Image.fromarray(current_image) for current_image in current_images]

            # Encodes action choices in the metadata
            metadata = []
            for current_action, current_encoded_mu in zip(current_actions, current_encoded_mus):
                metadata.append({
                    "model": "ours",
                    "inferred_action": current_action,
                    "encoded_action": current_encoded_mu
                })
            # No information for the last sample
            metadata.append({
                "model": "ours"
            })
            # Creates the current video
            current_video = Video()
            current_video.add_content(current_images, [0] * sequence_length, [0] * sequence_length, metadata,
                                      [False] * sequence_length)
            all_videos.append(current_video)

        return all_videos

    def create_dataset(self, path, videos: List[Video], extension="png"):
        '''
        Creates a dataset with the given video sequences

        :param path: path where to save the dataset
        :param videos: list of the videos to save
        :return:
        '''

        for idx, video in enumerate(videos):
            current_path = os.path.join(path, f"{idx:05d}")
            video.save(current_path, extension)

    def check_and_normalize_range(self, observations: torch.Tensor) -> torch.Tensor:
        '''
        If the range of the observations is in [-1, 1] instead of [0, 1] it normalizes it
        :param observations: arbitrarily shaped tensor to normalize
        :return: the input tensor normalized in [0, 1]
        '''

        minimum_value = torch.min(observations).item()

        # Check range and normalize
        if minimum_value < 0:
            observations = (observations + 1) / 2
        return observations


def builder(config, dataset: VideoDataset, logger):
    return EvaluationDatasetBuilder(config, dataset, logger)
