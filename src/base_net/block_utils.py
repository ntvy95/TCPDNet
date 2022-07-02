# Credit to: https://github.com/NVlabs/MUNIT/blob/master/networks.py

import torch.nn as nn

def get_activation(activation):
    # initialize activation
    if activation == 'relu':
        activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        activation = nn.PReLU()
    elif activation == 'selu':
        activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'none':
        activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
    return activation
