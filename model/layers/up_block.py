import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    """
    Upsampling block.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, scale_factor=2, upscaling_mode="nearest", late_upscaling=False):
        '''

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param kernel_size: Size of the kernel
        :param padding: Size of padding
        :param scale_factor: Multiplicative factor such that output_res = input_res * scale_factor
        :param upscaling_mode: interpolation upscaling mode
        :param late_upscaling: if True upscaling is applied at the end of the block, otherwise it is applied at the beginning
        '''

        super(UpBlock, self).__init__()

        self.scale_factor = scale_factor
        self.upscaling_mode = upscaling_mode
        self.late_upscaling = late_upscaling
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):

        out = x
        # By default apply upscaling at the beginning
        if not self.late_upscaling:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.upscaling_mode)

        out = self.conv(out)
        out = self.norm(out)
        out = F.leaky_relu(out, 0.2)

        # If upscaling is required at the end, apply it afterwards
        if self.late_upscaling:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode=self.upscaling_mode)

        return out