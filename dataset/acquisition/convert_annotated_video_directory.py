import subprocess
import glob
import os
import pandas as pd
import cv2
import shutil
import multiprocessing as mp
from pathlib import Path

from PIL import Image

from dataset.video import Video


video_extension = "mp4"
frames_extension = "png"
root_directory = "tmp"
output_directory = "tmp/tennis_ours"
annotations_filename = "dataset/acquisition/tennis_annotations/annotations.csv"
frameskip = 0
#frameskip = 4
processes = 8

target_size = [256, 96]

def acquire_sequence(video_capture: cv2.VideoCapture, capture_index: int, sequence_data, output_path: str):
    '''
    Acquires the video sequence specified by sequence data from the video_capture_stream and saves it to output_path
    :param video_capture: video capture object representing the current input video
    :param capture_index: index of the next frame that will be read from the video capture
    :param sequence_data: (original_filename, begin_frame, end_frame, box top, box left, box bottom, box right) specifying
                          the sequence to acquire
    :param output_path: path at which to save the captured sequence
    :return: next index that will be read from the video_capture object
    '''

    if not video_capture.isOpened():
        raise Exception("VideoCapture object is not open")

    _, begin_frame, end_frame, top, left, bottom, right = sequence_data

    if capture_index > begin_frame:
        raise Exception(f"The current capture position {capture_index} succeeds the beginning of the sequence to acquire {begin_frame}\n"
                        f"Ensure that sequences in the same video are ordered by indexes and not overlapping")

    # Seeks to the beginning of the sequence
    while capture_index < begin_frame:
        _, _ = video_capture.read()
        capture_index += 1

    assert(capture_index == begin_frame)

    images = []
    while capture_index <= end_frame:
        read_correctly, frame = video_capture.read()
        # Checks end of video
        if not read_correctly:
            break
        capture_index += 1

        # Converts frame to rgb and creates PIL image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_image = Image.fromarray(frame).crop((left, top, right, bottom)).resize(target_size, Image.BICUBIC)
        images.append(current_image)

        # Skip the specified number of frames between frames to acquire
        skipped_frames = 0
        while skipped_frames < frameskip and capture_index <= end_frame:
            read_correctly, _ = video_capture.read()
            # Checks end of video
            if not read_correctly:
                break
            skipped_frames += 1
            capture_index += 1

    frames_count = len(images)
    actions = [None] * frames_count
    rewards = [None] * frames_count
    dones = [None] * frames_count
    metadata = [None] * frames_count

    # Saves the acquired video in the dataset format
    acquired_video = Video()
    acquired_video.add_content(images, actions, rewards, metadata, dones)
    acquired_video.save(output_path, frames_extension)

    return capture_index


def acquire_video(args):
    annotations, begin_idx = args
    annotations = annotations.sort_values("begin_frame")

    opened_video_filename = os.path.join(root_directory, annotations.iloc[0]["original_filename"])
    video_capture = cv2.VideoCapture(opened_video_filename)
    capture_index = 0

    # Elaborates all the sequences
    for sequence_idx in range(len(annotations)):

        # If the current sequence is from a new video, open it
        current_video_filename = os.path.join(root_directory, annotations.iloc[sequence_idx]["original_filename"])
        if current_video_filename != opened_video_filename:
            video_capture = cv2.VideoCapture(current_video_filename)
            capture_index = 0
            opened_video_filename = current_video_filename

        print(f"- Acquiring sequence {sequence_idx} in '{current_video_filename}'")

        sequence_data = tuple(annotations.iloc[sequence_idx].to_numpy()[1:])  # Discards the index
        output_path = os.path.join(output_directory, f"{sequence_idx + begin_idx:05d}")
        capture_index = acquire_sequence(video_capture, capture_index, sequence_data, output_path)


if __name__ == "__main__":

    # Reads the video annotations
    annotations = pd.read_csv(annotations_filename)

    # Creates the output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    dataframes = annotations.groupby('original_filename')
    dataframes = [current_element[1] for current_element in dataframes] # Extracts the dataframe objects

    work_items = []
    begin_index = 0
    for dataframe in dataframes:
        work_items.append((dataframe, begin_index))
        begin_index += len(dataframe)

    pool = mp.Pool(processes)
    pool.map(acquire_video, work_items)
    pool.close()



























