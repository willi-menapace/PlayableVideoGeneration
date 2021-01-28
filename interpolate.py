import argparse
import importlib
import os
import shutil
import time
import random

import torch
import torchvision
import numpy as np
import cv2 as cv
import pickle
from PIL import Image

from dataset.dataset_splitter import DatasetSplitter
from dataset.transforms import TransformsGenerator
from dataset.video_dataset import VideoDataset
from evaluation.evaluator import Evaluator
from training.trainer import Trainer
from utils.configuration import Configuration
from utils.input_helper import InputHelper
from utils.logger import Logger
from utils.save_video_ffmpeg import VideoSaver

save_directory = "interpolation_results"
image_extension = "png"
interpolation_steps_count = 10
frames_count = 10
zoom_factor = 10
framerate = 5

if __name__ == "__main__":
    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    arguments = parser.parse_args()

    config_path = arguments.config

    configuration = Configuration(config_path)
    configuration.check_config()
    configuration.create_directory_structure()

    config = configuration.get_config()

    logger = Logger(config)
    search_name = config["model"]["architecture"]
    model = getattr(importlib.import_module(search_name), 'model')(config)
    model.cuda()

    datasets = {}

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config, split = dataset_splits[key]
        transform = transformations[key]

        datasets[key] = VideoDataset(path, batching_config, transform, split)

    trainer = Trainer(config, model, datasets["train"], logger)
    evaluator = Evaluator(config, datasets["validation"], logger, action_sampler=None, logger_prefix="validation")

    # Resume training
    try:
        trainer.load_checkpoint(model)
    except Exception as e:
        logger.print(e)
        logger.print("Cannot play without loading checkpoint")
        exit(1)

    model.eval()
    dataloader = evaluator.dataloader # Uses validation dataloader
    #dataset_index = int(input(f"- Insert start sample index in [0, {len(dataloader)}): "))
    dataset_index = 0


    # Erases and creates the new directory
    print(f"- Erasing '{save_directory}'")
    if os.path.isdir(save_directory):
        shutil.rmtree(save_directory)
    os.mkdir(save_directory)
    current_sequence_idx = 0
    """if not os.path.exists(save_directory):
        os.mkdir(save_directory)
        current_sequence_idx = 0
    else:
        directories = sorted(os.listdir(save_directory))
        if len(directories) == 0:
            current_sequence_idx = 0
        else:
            current_sequence_idx = directories[0]"""

    video_saver = VideoSaver()


    # Gets the first batch
    for current_batches in dataloader:
        break

    with torch.no_grad():
        observation_batch = current_batches.to_tuple()[0]

        # Samples the starting index
        batch_size = observation_batch.size(0)
        observations_count = observation_batch.size(1)
        batch_idx = 0
        observation_idx = 0

        first_action = int(input("Insert first action: "))
        second_action = int(input("Insert second action: "))

        interpolation_values = np.linspace(0.0, 1.0, interpolation_steps_count + 1).tolist()

        for current_interpolation_value in interpolation_values:

            # Creates the output directory
            current_output_directory = os.path.join(save_directory, str(current_sequence_idx))
            current_metadata_filename = os.path.join(current_output_directory, "play_metadata.pkl")
            video_filename = os.path.join(current_output_directory, "video.mp4")
            os.mkdir(current_output_directory)
            print(f"- Saving output to '{current_output_directory}'")

            current_frame_idx = 0
            model.start_inference()

            current_observation = observation_batch[batch_idx, observation_idx]  # Extract the first batch element and the first observation in the sequence
            current_frame = current_observation[:3]

            frames = []  # Generated frames for the current sequence
            frame_timestamps = []  # Timestamps at which each frame started being visualized on the screen
            actions = []  # Sequence of actions used to generate the current sequence

            while True:
                # Display the frame
                permuted_frame = (((current_frame + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                frames.append(permuted_frame)
                pil_image = Image.fromarray(permuted_frame)
                pil_image.save(os.path.join(current_output_directory, f"{current_frame_idx}.{image_extension}"))

                # Request exit
                if current_frame_idx == frames_count:

                    frames = np.stack(frames, axis=0)
                    video_saver.save_video(frames, video_filename, framerate)
                    # Restarts the game
                    break

                actions.append(0) # Saves the current action
                current_frame, current_observation = model.generate_next_interpolation(current_observation, first_action, second_action, current_interpolation_value)

                # Frame cycle end
                current_frame_idx += 1

            # Sequence cycle end
            current_sequence_idx += 1



