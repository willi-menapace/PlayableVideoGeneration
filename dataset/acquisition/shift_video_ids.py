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

root_directory = "/home/willi/dl/animation/video-generation/data/breakout_v2/test"


if __name__ == "__main__":
    '''
    Shifts the names of the indexes of the directories by a certain amount
    '''

    current_output_idx = 0

    if not os.path.isdir(root_directory):
        raise Exception(f"The folder '{root_directory}' does not exist")

    # Gets all directories
    video_paths_list = sorted(glob.glob(os.path.join(root_directory, "*")))
    video_paths_list = [current_path for current_path in video_paths_list if os.path.isdir(current_path)]

    current_idx = 0
    for current_video_path in video_paths_list:

        directory_name = os.path.dirname(current_video_path)

        target_name = os.path.join(directory_name, f"{current_idx:05d}")
        print(f"Renaming '{current_video_path}' to '{target_name}'")
        os.rename(current_video_path, target_name)

        current_idx += 1























