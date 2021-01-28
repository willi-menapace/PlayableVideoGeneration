import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
from utils.tensor_displayer import TensorDisplayer


class BreakoutPlatformPosition(nn.Module):

    def __init__(self):
        super(BreakoutPlatformPosition, self).__init__()

        platform_color = [200, 72, 72]
        platform_color_lower = [100, 72, 72]
        # Creates upper and lower bounds. They have three dimensions to align to the channels in the C, H, W tensor format
        self.lower_color_bound = torch.tensor(platform_color_lower, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).cuda() / 255 - 0.15
        self.upper_color_bound = torch.tensor(platform_color, dtype=torch.float).unsqueeze(-1).unsqueeze(-1).cuda() / 255 + 0.15

        self.positions = None
        self.platform_row_scale = 188 / 208 # Vertical position of the platform

    def create_positions_mask(self, height, width):
        '''
        Creates a (height, width) tensor whose value in each point is the x coordinate

        :param height: The height of the tensor to create
        :param width: The width of the tensor to create
        :return:
        '''

        mask = torch.arange(width).unsqueeze(0).cuda()
        mask = mask.repeat(height, 1)

        # Do not detect things above the platform
        upper_limit = int(187 / 208 * height) + 1
        mask[:upper_limit] = 0

        self.platform_row = int(self.platform_row_scale * height)

        return mask

    def detect_platform(self, frame: np.ndarray) -> int:
        '''
        Computes the position of the lower left part of the platform

        :param frame: (channels, height, width) boolean tensor with True in the positions of the frame where the
                                                platform color is detected
        :return: the x position of the left platform edge, -1 if none was found
        '''

        width = frame.shape[-1]
        current_position_length = 0
        current_start_position = 0
        for idx in range(width):
            if frame[0, self.platform_row, idx] == True and idx != (width - 1):
                if current_position_length == 0:
                    current_start_position = idx
                current_position_length += 1
            else:
                if current_position_length > 0:
                    if current_position_length > 11:
                        return current_start_position
                    current_position_length = 0

        return -1

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes the position of the lower left part of the platform

        :param observations: (bs, observations_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with x positions of the player-controlled bar
        '''

        batch_size = observations.size(0)
        observations_count = observations.size(1)
        channels = observations.size(2)
        height = observations.size(3)
        width = observations.size(4)

        # If no position mask has been previously created, create one with the dimensions of the batch
        if self.positions is None:
            positions_mask = self.create_positions_mask(height, width)
            positions_mask = positions_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            positions_mask = positions_mask.repeat(1, observations_count, channels, 1, 1) # Batch size may vary in the future, so do not repeat
            self.positions = positions_mask

        # Expand the positions mask to the current batch size
        current_positions_mask = self.positions.repeat(batch_size, 1, 1, 1, 1)
        # Individuates the platform
        platform_mask = torch.ge(observations, self.lower_color_bound) & torch.le(observations, self.upper_color_bound)

        platform_mask = platform_mask.cpu().numpy()


        # Scans the row of the platform in search of elements of the color of the platform
        all_positions = []
        for sequence_idx in range(batch_size):
            current_positions = []
            for observation_idx in range(observations_count):
                current_frame = platform_mask[sequence_idx, observation_idx]
                current_position = self.detect_platform(current_frame)
                current_positions.append(current_position)
            all_positions.append(current_positions)
        all_positions = np.asarray(all_positions)

        #platform_positions = (current_positions_mask * platform_mask)
        #platform_positions[platform_positions == 0] = 100000
        #platform_positions = platform_positions.reshape(batch_size, observations_count, -1).max(dim=2)[1]
        return all_positions
