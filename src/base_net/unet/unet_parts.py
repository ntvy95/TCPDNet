""" Parts of the U-Net model """
# A modification of: https://github.com/milesial/Pytorch-UNet
# with a heavy inspiration from: https://github.com/NVlabs/MUNIT/

# on 20210326 modified by mtanaka@sc.e.titech.ac.jp

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_net.conv2dblock import Conv2dBlock

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3,
                    stride=1, dilation=1, padding='same', norm='none', activation='relu', pad_type='zero', bias=True, groups=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            Conv2dBlock( in_channels, mid_channels, kernel_size, stride, dilation, padding, norm, activation, pad_type, bias, groups),
            Conv2dBlock(mid_channels, out_channels, kernel_size, stride, dilation, padding, norm, activation, pad_type, bias, groups),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                    stride=1, dilation=1, padding='same', norm='none', activation='relu', pad_type='zero', bias=True, groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, None, kernel_size, stride, dilation, padding, norm, activation, pad_type, bias, groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, upsampling='bilinear', kernel_size=3,
                    stride=1, dilation=1, padding='same', norm='none', activation='relu', pad_type='zero', bias=True, groups=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if upsampling == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size, stride, dilation, padding, norm, activation, pad_type, bias, groups)
        elif upsampling == 'transpose_conv':
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, None, kernel_size, stride, dilation, padding, norm, activation, pad_type, bias, groups)
        else:
            assert 0, "Unsupporting upsampling type: {}".format(upsampling)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2.shape[-1] - x1.shape[-1] == 1:
            x1 = F.pad(x1, (0, 1), 'constant', 0)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpNoSkip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, upsampling='bilinear', kernel_size=3,
                    stride=1, dilation=1, padding='same', norm='none', activation='relu', pad_type='zero', bias=True, groups=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if upsampling == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size, stride, dilation, padding, norm, activation, pad_type, bias, groups)
        elif upsampling == 'transpose_conv':
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels, None, kernel_size, stride, dilation, padding, norm, activation, pad_type, bias, groups)
        else:
            assert 0, "Unsupporting upsampling type: {}".format(upsampling)


    def forward(self, x):
        return self.conv(self.up(x))

