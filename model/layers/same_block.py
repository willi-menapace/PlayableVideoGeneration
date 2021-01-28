import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SameBlock(nn.Module):
    '''
    Convolutional block with normalization and activation
    '''
    expansion = 1

    def __init__(self, in_planes, out_planes, downsample_factor=1, drop_final_activation=False):
        '''

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param downsample_factor: Reduction factor in feature dimension
        :param drop_final_activation: if True does not pass the final output through the activation function
        '''

        super(SameBlock, self).__init__()

        norm_layer = nn.BatchNorm2d

        self.downsample_factor = downsample_factor
        self.drop_final_activation = drop_final_activation

        self.conv1 = conv3x3(in_planes, out_planes, stride=1)
        self.bn1 = norm_layer(out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        # Downscale if required
        if self.downsample_factor != 1:
            out = F.avg_pool2d(out, self.downsample_factor)
        out = self.bn1(out)
        # Applies activation if required
        if not self.drop_final_activation:
            out = self.relu(out)

        return out
