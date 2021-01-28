import subprocess
import glob
import os
import pandas as pd
import cv2
import shutil
import multiprocessing as mp
from pathlib import Path
from distutils.dir_util import copy_tree

from dataset.video import Video

root_directory = "/home/willi/dl/animation/video-generation/data/breakout_v2/train"
output_directory = "/home/willi/dl/animation/video-generation/data/breakout_v2/variable_length_train"
frame_skip = 0
sequence_length = -1
extension = "png"

target_size = [160, 208]
crop_region = [0, 1, 160, 209]

if __name__ == "__main__":
    '''
    Subsamples videos in a dataset and makes them fixed length
    '''

    current_output_idx = 0
    # Creates the output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Gets all directories
    video_paths_list = sorted(glob.glob(os.path.join(root_directory, "*")))
    video_paths_list = [current_path for current_path in video_paths_list if os.path.isdir(current_path)]

    for current_video_path in video_paths_list:

        print(f"- Splitting sequence '{current_video_path}'")
        # Split the video
        current_video = Video()
        current_video.load(current_video_path)
        video_splits = current_video.subsample_split_resize(frame_skip, sequence_length, crop_region, target_size)
        print(f"  - Sequence split to {len(video_splits)} sequences'")

        # Save each output sequence
        for current_split in video_splits:
            output_path = os.path.join(output_directory, f"{current_output_idx:05d}")
            current_output_idx += 1

            print(f"  - Saving split to '{output_path}'")
            current_split.save(output_path, extension=extension)























