import torch.nn as nn
from base_net import *
from net_builders import build_base_net_configs
import torch

def is_valid_net(net_name):
    return net_name in ['Conv2dBlock', 'UNet']

class RefineNet(nn.Module):
    def __init__(self, net, in_channels, out_channels, skip_connection_flag, setting):
        super(RefineNet, self).__init__()
        setting = build_base_net_configs(net, setting)
        assert is_valid_net(setting['name'])
        self.net = globals()[setting['name']](in_channels, out_channels, setting)
        self.skip_connection_flag = skip_connection_flag

        # print our configuration
        print(f"RefineNet: {setting['name']}")
        print(f"skip_connection_flag: {self.skip_connection_flag}")

    def forward(self, x):
        y = self.net(x)
        if self.skip_connection_flag:
            y = x + y
        return y
