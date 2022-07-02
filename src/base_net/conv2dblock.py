# Credit to: https://github.com/NVlabs/MUNIT/blob/master/networks.py
# on 20210326 modified by mtanaka@sc.e.titech.ac.jp

from .block_utils import *

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size,
                 stride=1, dilation=1, padding='same', norm='none', activation='relu', pad_type='zero', bias=True, groups=1):
        super(Conv2dBlock, self).__init__()

        if( isinstance(padding,str) and padding.lower() == 'same' ):
            padding = (kernel_size-1)//2

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.activation = get_activation(activation)
        
        # initialize convolution
        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=bias, groups=groups))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=bias, groups=groups)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if( self.norm is not None ):
            x = self.norm(x)
        if( self.activation is not None ):
            x = self.activation(x)
        return x

