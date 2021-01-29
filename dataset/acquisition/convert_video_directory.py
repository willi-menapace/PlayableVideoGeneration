#ffmpeg -i 000000.mp4 -filter:v fps=10  %05d.png
import argparse
import subprocess
import glob
import os
import shutil
import multiprocessing as mp
from pathlib import Path

from PIL import Image

from dataset.video import Video

def acquire_video_wrapper(args):
    '''
    Wraps the call to the acquire video functions
    :param args: arguments to unpack
    :return:
    '''

    return acquire_video(*args)

def acquire_video(video_path, output_path, tmp_path, fps, extension, target_size):
    '''
    Acquires a video and saves it to the specified output directory

    :param video_path: the video file path
    :param output_path: the directory in which to save the output video. The directory must not exist
    :param tmp_path: directory where to save temporary output
    :param fps: fps at which to acquire the video
    :param extension: extension in which to save files
    :param target_size: (width, height) at which to save the output frames
    :return:
    '''

    print(f" - Acquiring '{video_path}'")

    # Cleans the tmp_directory
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)

    os.mkdir(tmp_path)

    # Decodes the video frames
    subprocess.run(["ffmpeg", "-i", video_path, "-filter:v", f"fps={fps}", f"{tmp_path}/%05d.{extension}"])

    # Acquires the ordered names of the generated frames
    frame_paths = list(sorted(glob.glob(os.path.join(tmp_path, f"*.{extension}"))))
    frames_count = len(frame_paths)

    # Checks that frames were generated
    if frames_count <= 0:
        raise Exception(f"Reading video '{video_path}', but no frames were generated")

    # Reads the generated images
    images = [Image.open(current_frame_path) for current_frame_path in frame_paths]
    images = [current_image.resize(target_size, Image.BICUBIC) for current_image in images]

    actions = [None] * frames_count
    rewards = [None] * frames_count
    dones = [None] * frames_count
    metadata = [None] * frames_count

    # Saves the acquired video in the dataset format
    acquired_video = Video()
    acquired_video.add_content(images, actions, rewards, metadata, dones)
    acquired_video.save(output_path, extension)

    # Clears temporary files
    shutil.rmtree(tmp_path)


if __name__ == "__main__":

    print("== Video Search ==")

    # Loads arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--target_size", type=int, nargs=2, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--output_extension", type=str, default="png")

    arguments = parser.parse_args()

    video_extension = arguments.video_extension
    output_extension = arguments.output_extension
    root_directory = arguments.video_directory
    output_directory = arguments.output_directory
    acquisition_fps = arguments.fps
    target_size = arguments.target_size
    processes = arguments.processes

    # Searches the top level directories
    directories_to_process = [current_file for current_file in glob.glob(os.path.join(root_directory, "*")) if os.path.isdir(current_file)]
    directories_to_process.append(root_directory)
    directories_to_process.sort()

    # Extracts the paths of all videos in the directories to process
    video_paths = []
    for current_directory in directories_to_process:
        directory_contents = glob.glob(os.path.join(current_directory, f"*.{video_extension}"))
        video_paths.extend(directory_contents)
    video_paths.sort()

    # Creates the output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    output_paths = [os.path.join(output_directory, f"{index:05d}") for index in range(len(video_paths))]
    tmp_paths = [os.path.join(output_directory, f"tmp_{index:05d}") for index in range(len(video_paths))]

    # List worker parameters
    work_items = list(zip(video_paths, output_paths, tmp_paths, [acquisition_fps] * len(video_paths),
                          [output_extension] * len(video_paths), [target_size] * len(video_paths)))

    print("== Video Acquisition ==")

    # Processes all videos
    pool = mp.Pool(processes)
    pool.map(acquire_video_wrapper, work_items)
    pool.close()

    print("== Done ==")


























