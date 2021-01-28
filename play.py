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

save_directory = "play_results"
image_extension = "png"
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
    input_helper = InputHelper(interactive=False)
    window_name = "rendered_frame"
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    while True:
        # Gets the first batch
        for current_batches in dataloader:
            break


        # Creates the output directory
        current_output_directory = os.path.join(save_directory, str(current_sequence_idx))
        current_metadata_filename = os.path.join(current_output_directory, "play_metadata.pkl")
        video_filename = os.path.join(current_output_directory, "video.mp4")
        video_timecoded_filename = os.path.join(current_output_directory, "video_timecoded.mp4")
        video_actions_filename = os.path.join(current_output_directory, "video_actions.mp4")
        video_timecoded_actions_filename = os.path.join(current_output_directory, "video_timecoded_actions.mp4")
        os.mkdir(current_output_directory)
        print(f"- Saving output to '{current_output_directory}'")

        with torch.no_grad():
            observation_batch = current_batches.to_tuple()[0]

            # Samples the starting index
            batch_size = observation_batch.size(0)
            observations_count = observation_batch.size(1)
            batch_idx = random.randint(0, batch_size - 1)
            observation_idx = random.randint(0, observations_count - 1)

            current_observation = observation_batch[batch_idx, observation_idx] # Extract the first batch element and the first observation in the sequence
            current_frame = current_observation[:3]
            #current_state = model.get_state(initial_observation)
            #current_frame = model.decode_state(current_state)

            current_frame_idx = 0
            model.start_inference()


            frames = [] # Generated frames for the current sequence
            frame_timestamps = [] # Timestamps at which each frame started being visualized on the screen
            actions = [] # Sequence of actions used to generate the current sequence
            begin_time = 0
            current_action = None
            while True:
                # Display the frame
                permuted_frame = (((current_frame + 1) / 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                color_corrected_frame = np.copy(permuted_frame)
                color_corrected_frame[:, :, 0] = permuted_frame[:, :, 2]
                color_corrected_frame[:, :, 2] = permuted_frame[:, :, 0]
                display_frame = cv.resize(color_corrected_frame, (color_corrected_frame.shape[1] * zoom_factor, color_corrected_frame.shape[0] * zoom_factor))
                if current_action is not None:
                    display_frame = video_saver.draw_text_on_frame(display_frame, (40, 20), str(current_action + 1), pointsize=128)
                cv.imshow(window_name, display_frame)
                #cv.waitKey(1)

                # Start the timer at the first frame
                if begin_time == 0:
                    begin_time = time.time()
                    frame_time = 0
                # At subsequent frames use the current time
                else:
                    frame_time = time.time() - begin_time
                frame_timestamps.append(frame_time)

                frames.append(permuted_frame)
                pil_image = Image.fromarray(permuted_frame)
                pil_image.save(os.path.join(current_output_directory, f"{current_frame_idx}.{image_extension}"))

                # Asks for input until a correct one is received
                success = False
                while not success:
                    success = False
                    try:
                        print(f"\n- Insert current action in [1, {config['data']['actions_count']}], 0 to reset: ")
                        #current_action = int(input_helper.read_character())
                        current_action = int(cv.waitKey(0)) - ord('0')

                        current_action -= 1 # Puts the action in the expected range for the model
                        if current_action != -1 and (current_action < 0 or current_action >= config['data']['actions_count']):
                            success = False
                        else:
                            success = True
                    except Exception as e:
                        time.sleep(0.1)
                        success = False

                # Request exit
                if current_action == -1:
                    # Saves metadata
                    metadata = {
                        "actions": actions,
                        "timestamps": frame_timestamps
                    }
                    with open(current_metadata_filename, "wb") as file:
                        pickle.dump(metadata, file)

                    frames = np.stack(frames, axis=0)
                    video_saver.save_video(frames, video_filename, framerate)
                    video_saver.save_timecoded_video(frames, frame_timestamps, video_timecoded_filename, framerate)
                    video_saver.save_action_video(frames, actions, video_actions_filename, framerate)
                    video_saver.save_timecoded_action_video(frames, actions, frame_timestamps, video_timecoded_actions_filename, framerate)

                    # Restarts the game
                    break

                actions.append(current_action + 1) # Saves the current action
                current_frame, current_observation = model.generate_next(current_observation, current_action)

                #current_state = model.next_state(current_state, current_action)
                #current_frame = model.decode_state(current_state)

                # Frame cycle end
                current_frame_idx += 1

        # Sequence cycle end
        current_sequence_idx += 1



