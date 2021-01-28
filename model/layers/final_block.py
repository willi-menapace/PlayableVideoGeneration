import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return

class FinalBlock(nn.Module):
    '''
    Final block transforming features into an image
    '''

    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1):
        '''

        :param in_features: Input features to the module
        :param out_features: Output feature
        '''

        super(FinalBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=True)

    def forward(self, x):

        x = self.conv(x)
        x = torch.tanh(x)

        return x
