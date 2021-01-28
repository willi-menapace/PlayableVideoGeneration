import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.residual_block import ResidualBlock


class RepresentationNetwork(nn.Module):
    '''
    Model that encodes an observation into a state with action attention
    '''

    def __init__(self, config):
        super(RepresentationNetwork, self).__init__()

        self.config = config
        self.in_features = self.config["training"]["batching"]["observation_stacking"] * 3

        self.conv1 = nn.Conv2d(self.in_features, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        residual_blocks = [
            ResidualBlock(16, 16, downsample_factor=1),  # res / 2
            ResidualBlock(16, 32, downsample_factor=2),  # res / 4
            ResidualBlock(32, 32, downsample_factor=1),  # res / 4
            ResidualBlock(32, 64, downsample_factor=2),    #res / 8
            ResidualBlock(64, 64, downsample_factor=1),    #res / 8
            ResidualBlock(64, 64 + 1, downsample_factor=1),  # res / 8
        ]
        self.residuals = nn.Sequential(*residual_blocks)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes the state corresponding to each observation

        :param observations: (bs, 3 * observation_stacking, height, width) tensor
        :return: (bs, states_features, states_height, states_width) tensor of states
                 (bs, 1, states_height, states_width) tensor with attention
        '''

        x = self.conv1(observations)
        x = F.avg_pool2d(x, 2)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.residuals(x)

        # Separates the last layer which represents the attention map
        state = x[:, :-1]

        # Computes the softmax over the spatial dimentions (width, height)
        attention = x[:, -1:]
        attention_shape = attention.size()
        attention_flat_shape = [attention_shape[0], attention_shape[1], attention_shape[2] * attention_shape[3]]
        flat_attention = attention.reshape(attention_flat_shape)
        flat_attention = torch.sigmoid(flat_attention)
        attention = flat_attention.reshape(attention_shape)

        return state, attention