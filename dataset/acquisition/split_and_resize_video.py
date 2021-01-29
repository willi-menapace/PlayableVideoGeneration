import math
import subprocess
import glob
import os
import shutil
import multiprocessing as mp
from pathlib import Path

from PIL import Image

# ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 input.mp4
# ffmpeg -i tennis_djokovic_federed_wimbledon.mp4 -ss 0 -t 600 -s 426x240 -r 5 fast_resized_00001.mp4
# ffmpeg -ss 00:00:00 -t 00:10:00 -i tennis_djokovic_federed_wimbledon.mp4 -acodec copy -vcodec copy 00001.mp4

# Directory in which to search files
root_directory = "tmp"
video_extension = "mp4"

# Length in seconds of each split
splits_length = 3600

# Framerate for the reduced split versions
reduced_splits_framerate = 5
# Resolution for the reduces split versions
reduced_splits_resolution = [426, 240]

processes = 8

def get_video_duration(filename: str) -> float:
    '''
    Returns the duration in seconds of the specified video
    :param filename: video file of which to compute the duration
    :return:
    '''

    pipe = subprocess.Popen(["ffprobe", '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', filename], stdout=subprocess.PIPE).stdout
    output = pipe.read()
    duration = float(output)
    return duration


def create_segment(params):

    filename, split_idx = params
    print(f"- Creating segment {split_idx} in '{filename}'")

    extension_length = len(filename.split(".")[-1]) + 1
    unextended_filename = filename[:-extension_length]
    output_directory = unextended_filename + "_splits"
    base_video_name = unextended_filename.split("/")[-1]

    current_output_split = os.path.join(output_directory, f"{base_video_name}_{split_idx:05d}.{video_extension}")
    current_output_reduced_split = os.path.join(output_directory, f"{base_video_name}_reduced_{split_idx:05d}.{video_extension}")

    current_begin_time = split_idx * splits_length

    command_parameters = ["ffmpeg", '-ss', f'{current_begin_time}', '-t', f'{splits_length}', "-i", filename,
                          '-acodec', 'copy', '-vcodec', 'copy', current_output_split]

    subprocess.run(command_parameters)


def split_video(filename: str, pool: mp.Pool):
    '''
    Splits the given video according to the global parameters
    :param filename: video file to split
    :param pool: pool to use for processing
    :return:
    '''

    # Creates the output directory
    extension_length = len(filename.split(".")[-1]) + 1
    unextended_filename = filename[:-extension_length]
    output_directory = unextended_filename + "_splits"
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Computes the number of splits
    video_length = get_video_duration(filename)
    splits_count = math.ceil(video_length / splits_length)

    # Creates parameters for multiprocessing workers
    work_items = []
    for split_idx in range(splits_count):
        work_items.append((filename, split_idx))

    pool.map(create_segment, work_items)



if __name__ == "__main__":

    videos = list(glob.glob(os.path.join(root_directory, f'*.{video_extension}')))
    pool = mp.Pool(processes)

    # Splits all the videos
    for current_video in videos:
        print(f"== Processing video '{current_video}' ==")
        split_video(current_video, pool)

    pool.close()