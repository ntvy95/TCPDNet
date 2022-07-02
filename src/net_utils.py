import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def sub_sample(x):
    # dim info of x: (batch, 1, h, w)
    tl = x[:,:,0::2,0::2]
    tr = x[:,:,0::2,1::2] 
    bl = x[:,:,1::2,0::2] 
    br = x[:,:,1::2,1::2] 
    return torch.stack([tl, tr, bl, br], axis=1)

def sub_image(x, mask):
    # dim info of x: (batch, 1, h, w)
    # dim info of mask: (c, h, w)
    h, w = x.shape[-2:]
    n_channel = mask.shape[0]
    y = torch.zeros(x.shape[0], n_channel, h, w).to(device=x.device, dtype=x.dtype)
    mask = mask[None, :, :h, :w]
        
    x = x.squeeze(1)
    for idx in range(n_channel):
        y[:, idx, :, :] = x*mask[:, idx, :, :]
    return y


def bilinear_interp(x, is_green=False, freq=4):
    # dim info of x: (batch, c, h, w)
    if is_green:
        if freq == 4:
            w = torch.tensor([[[[0, 1/4, 0],
                                [1/4, 1, 1/4],
                                [0, 1/4, 0]]]], dtype=x.dtype).to(device=x.device)
        elif freq == 16:
            w = torch.tensor([[[[0, 1/4, 1/4, 1/4, 0],
                                [1/4, 1/2, 1/2, 1/2, 1/4],
                                [1/4, 1/2, 1, 1/2, 1/4],
                                [1/4, 1/2, 1/2, 1/2, 1/4],
                                [0, 1/4, 1/4, 1/4, 0]]]], dtype=x.dtype).to(device=x.device)
    else:
        if freq == 4:
            w = torch.tensor([[[[1/4, 1/2, 1/4],
                                [1/2, 1, 1/2],
                                [1/4, 1/2, 1/4]]]], dtype=x.dtype).to(device=x.device)
        elif freq == 16:
            w = torch.tensor([[[[1/16, 1/8, 3/16, 1/4, 3/16, 1/8, 1/16],
                                [1/8, 1/4, 3/8, 1/2, 3/8, 1/4, 1/8],
                                [3/16, 3/8, 9/16, 3/4, 9/16, 3/8, 3/16],
                                [1/4, 1/2, 3/4, 1, 3/4, 1/2, 1/4],
                                [3/16, 3/8, 9/16, 3/4, 9/16, 3/8, 3/16],
                                [1/8, 1/4, 3/8, 1/2, 3/8, 1/4, 1/8],
                                [1/16, 1/8, 3/16, 1/4, 3/16, 1/8, 1/16]]]], dtype=x.dtype).to(device=x.device)

    w.detach_()
    c = x.shape[1]
    w = w.repeat(c,1,1,1)
    if freq == 4:
        x = F.pad(x, (1, 1, 1, 1), "reflect")
        #x = F.pad(x, (1, 1, 1, 1), "constant")
    elif freq == 16: # demands manual padding
        x = extended_reflective_padding(x, is_green)
        #x = F.pad(x, (2, 2, 2, 2), "constant") if is_green else F.pad(x, (3, 3, 3, 3), "constant") 
    return F.conv2d(x, w, bias=None, groups=c)

def extended_reflective_padding(x, is_green=False):
    if is_green:
        y = torch.zeros(x.shape[0], x.shape[1], x.shape[2]+4, x.shape[3]+4, dtype=x.dtype, device=x.device)
        y[:,:,2:-2,2:-2] = x
        #tl
        y[:,:,:2,:2] = x[:,:,2:4,2:4]
        #bl
        y[:,:,-2:,-2:] = x[:,:,-4:-2,-4:-2]
        #ml
        y[:,:,2:-2,:2] = x[:,:,:,2:4]
        #mr
        y[:,:,2:-2,-2:] = x[:,:,:,-4:-2]
        #top
        y[:,:,:2,2:-2] = x[:,:,2:4,:]
        #bottom
        y[:,:,-2:,2:-2] = x[:,:,-4:-2,:]
        #tr
        y[:,:,:2,-2:] = x[:,:,2:4,-4:-2]
        #br
        y[:,:,-2:,:2] = x[:,:,-4:-2,2:4]
        x = y
    else: 
        y = torch.zeros(x.shape[0], x.shape[1], x.shape[2]+6, x.shape[3]+6, dtype=x.dtype, device=x.device)
        y[:,:,3:-3,3:-3] = x
        y[:,:,:3,3:-3] = x[:,:,1:4,:]
        y[:,:,-3:,3:-3] = x[:,:,-4:-1,:]
        y[:,:,3:-3,:3] = x[:,:,:,1:4]
        y[:,:,3:-3,-3:] = x[:,:,:,-4:-1]
        y[:,:,:3,:3] = x[:,:,1:4,1:4]
        y[:,:,-3:,-3:] = x[:,:,-4:-1,-4:-1]
        y[:,:,:3,-3:] = x[:,:,1:4,-4:-1]
        y[:,:,-3:,:3] = x[:,:,-4:-1,1:4]
        x = y
    return x

    
