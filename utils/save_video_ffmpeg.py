import subprocess as sp
from math import floor
from typing import List, Tuple
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


import numpy as np

class VideoSaver:

    def __init__(self):
        pass

    def draw_text_on_frame(self, frame: np.ndarray, point: Tuple[int], text: str, fillcolor="white", shadowcolor="black", pointsize=30):
        '''
        Draws the given text on a frame

        :param frame: (height, width, 3) uint8 array on which to draw
        :param point: (x, y) coordinates where to draw
        :param text: the text to draw
        :param fillcolor: internal color
        :param shadowcolor: border color
        :param pointsize: size of text
        :return: (height, width, 3) uint8 array
        '''

        image = Image.fromarray(frame)

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("utils/fonts/Roboto-Regular.ttf", pointsize)

        x, y = point
        # thin border
        draw.text((x - 1, y), text, font=font, fill=shadowcolor)
        draw.text((x + 1, y), text, font=font, fill=shadowcolor)
        draw.text((x, y - 1), text, font=font, fill=shadowcolor)
        draw.text((x, y + 1), text, font=font, fill=shadowcolor)

        # thicker border
        #draw.text((x - 1, y - 1), text, font=font, fill=shadowcolor)
        #draw.text((x + 1, y - 1), text, font=font, fill=shadowcolor)
        #draw.text((x - 1, y + 1), text, font=font, fill=shadowcolor)
        #draw.text((x + 1, y + 1), text, font=font, fill=shadowcolor)

        # now draw the text over it
        draw.text((x, y), text, font=font, fill=fillcolor)

        # Returns the image with the overlaid test
        return np.asarray(image)

    def write_actions_on_frames(self, frames: np.ndarray, actions: List[int]) -> np.ndarray:
        '''
        Saves a video where each frame in the sequence has the performed action impressed on it

        :param frames: (frames_count, height, width, 3) uint8 array with the frames
        :param actions: list of frames_count - 1 int values indicating the action performed at each frame, excluding the first one
        :return: (frames_count, height, width, 3) uint8 array with the frames with actions
        '''

        frames_count = frames.shape[0]
        actions_count = len(actions)
        if frames_count != actions_count + 1:
            raise Exception(f"Expected {frames_count - 1} actions, but {actions_count} were received")

        new_frames = [frames[0]]
        for frame_idx in range(1, frames_count):
            current_frame = frames[frame_idx]
            current_action = actions[frame_idx - 1]

            # Overlays the text on the image
            current_frame = self.draw_text_on_frame(current_frame, (10, 10), str(current_action), pointsize=16)

            new_frames.append(current_frame)

        new_frames = np.stack(new_frames, axis=0)
        return new_frames

    def save_action_video(self, frames: np.ndarray, actions: List[int], filename: str, framerate=30):
        '''
        Saves a video where each frame in the sequence has the performed action impressed on it

        :param frames: (frames_count, height, width, 3) uint8 array with the frames to save
        :param actions: list of frames_count - 1 int values indicating the action performed at each frame, excluding the first one
        :param filename: name for the output video
        :param framerate: framerate for the video to create
        :return:
        '''

        new_frames = self.write_actions_on_frames(frames, actions)
        # Saves the video
        self.save_video(new_frames, filename, framerate)

    def timecode_video(self, frames: np.ndarray, timestamps: List[float], framerate=30, last_frame_duration=0.2) -> np.ndarray:
        '''
        Creates a sequence of frames where each frame in the sequence appears at the specified timestamp

        :param frames: (frames_count, height, width, 3) uint8 array with the frames to timecode
        :param timestamps: list of frames_count float values indicating the second at which each frame should appear
        :param framerate: framerate for the video to create
        :param last_frame_duration: Duration in seconds of the last frame
        :return: (timecoded_frames_count, height, width, 3) uint8 array with the timecoded video
        '''

        frames_count = frames.shape[0]

        if len(timestamps) == 0 or len(timestamps) != frames_count:
            raise Exception(
                f"The length of timestamps ({len(timestamps)}) must match the number of frames ({frames_count})")
        if timestamps[0] != 0:
            raise Exception(
                f"The first frame must appear at time 0, but is specified to appear at time {timestamps[0]}")

        # Copies the timestamps so that they can be modified
        timestamps = timestamps[:]
        # Registers the duration of the last frame
        timestamps.append(timestamps[-1] + last_frame_duration)

        new_frames = []
        current_time = 0
        frame_duration = 1 / framerate
        for frame_idx in range(1, frames_count + 1):  # +1 to account for the added timestamp for the last frame
            # Computes how many frames must be produced to arrive at the next timestamp
            next_time = timestamps[frame_idx]
            needed_frame_repetitions = floor((next_time - current_time) / frame_duration)
            current_time = current_time + needed_frame_repetitions * frame_duration

            # Adds the required number of frames
            new_frames.extend([frames[frame_idx - 1]] * needed_frame_repetitions)

        new_frames = np.stack(new_frames, axis=0)
        return new_frames


    def save_timecoded_video(self, frames: np.ndarray, timestamps: List[float], filename: str, framerate=30, last_frame_duration=0.2):
        '''
        Saves a video where each frame in the sequence appears at the specified timestamp

        :param frames: (frames_count, height, width, 3) uint8 array with the frames to save
        :param timestamps: list of frames_count float values indicating the second at which each frame should appear
        :param filename: name for the output video
        :param framerate: framerate for the video to create
        :param last_frame_duration: Duration in seconds of the last frame
        :return:
        '''

        new_frames = self.timecode_video(frames, timestamps, framerate, last_frame_duration)

        # Saves the video
        self.save_video(new_frames, filename, framerate)

    def save_timecoded_action_video(self, frames: np.ndarray, actions: List[int], timestamps: List[float], filename: str, framerate=30, last_frame_duration=0.2):
        '''
        Saves a video where each frame in the sequence appears at the specified timestamp and with the specified action impressed on it

        :param frames: (frames_count, height, width, 3) uint8 array with the frames to save
        :param actions: list of frames_count - 1 int values indicating the action performed at each frame, excluding the first one
        :param timestamps: list of frames_count float values indicating the second at which each frame should appear
        :param filename: name for the output video
        :param framerate: framerate for the video to create
        :param last_frame_duration: Duration in seconds of the last frame
        :return:
        '''

        new_frames = self.write_actions_on_frames(frames, actions)
        new_frames = self.timecode_video(new_frames, timestamps, framerate, last_frame_duration)

        # Saves the video
        self.save_video(new_frames, filename, framerate)

    def save_video(self, frames: np.ndarray, filename: str, framerate=30):
        '''
        Converts a numpy array to videos

        :param frames: (frames_count, height, width, 3) uint8 array with the frames to save
        :param filename: name for the output video
        :param framerate: framerate for the video to create
        :return: 
        '''

        command = ["ffmpeg",
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', '{}x{}'.format(frames.shape[-2], frames.shape[-3]),
                   '-pix_fmt', 'rgb24',
                   '-r', str(framerate), # framerate
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-q:v', '3',
                   '-an', # do not expect audio
                   filename]

        pipe = sp.Popen(command, stdin=sp.PIPE)
        pipe.stdin.write(frames.tostring())
        pipe.stdin.close()


if __name__ == "__main__":

    frames_count = 5
    # Makes black and white frames in succession
    frames = np.zeros((frames_count, 208, 160, 3), np.uint8)
    for i in range(0, frames_count, 2):
        frames[i] += 255
    timestamps = [0, 1, 3, 7, 15]
    actions = [0, 1, 2, 3]

    video_saver = VideoSaver()
    video_saver.save_action_video(frames, actions, "/tmp/test.mp4", framerate=1)