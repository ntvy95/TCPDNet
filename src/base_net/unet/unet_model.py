""" Full assembly of the parts to form the complete network """
# A modification of: https://github.com/milesial/Pytorch-UNet
# inspired by: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/model.py
# on 20210326 modified by mtanaka@sc.e.titech.ac.jp

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, configs=None):
        super(UNet, self).__init__()
        channels = configs['channels']
        self.inc = DoubleConv(n_channels, channels[0], None, **configs['inc'])
        self.down = []
        in_channel = channels[0]
        for out_channel in channels[1:-1]:
            self.down.append(Down(in_channel, out_channel, **configs['down']))
            in_channel = out_channel
        factor = 2 if configs['up']['upsampling'] == "bilinear" else 1
        self.down.append(Down(in_channel, channels[-1] // factor, **configs['down']))
        self.down = nn.ModuleList(self.down)
        reversed_channels = list(reversed(channels))
        self.up = []
        in_channel = reversed_channels[0]
        for out_channel in reversed_channels[1:-1]:
            self.up.append(Up(in_channel, out_channel // factor, **configs['up']))
            in_channel = out_channel
        self.up.append(Up(channels[1], channels[0], **configs['up']))
        self.up = nn.ModuleList(self.up)
        self.outc = Conv2dBlock(channels[0], n_classes, **configs['outc'])

    def forward(self, x):
        x = self.inc(x)
        x_down = [x]
        for down in self.down:
            x = down(x)
            x_down.insert(0,x)
        x_down = x_down[1:]
        for up, xd in zip(self.up, x_down):
            x = up(x, xd)
        logits = self.outc(x)
        return logits
