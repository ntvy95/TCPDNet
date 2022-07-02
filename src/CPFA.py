import torch
import torch.nn as nn
from net_utils import sub_sample, sub_image

class CPFA:
    
    def __init__(self, color_order = "gbrg", max_height = 768, max_width = 1024, polar_order="90-45-135-0"):
        max_height = max_height + 16
        max_width = max_width + 16
        self.mask = torch.zeros(4, 3, max_height, max_width)
        if polar_order == "90-45-135-0":
            self.polar_order = [2, 1, 3, 0] #tl, tr, bl, br
        elif polar_order == "0-45-135-90":
            self.polar_order = [0, 1, 3, 2]
        elif polar_order == "0-135-45-90":
            self.polar_order = [0, 3, 1, 2]
        elif polar_order == "0-90-45-135":
            self.polar_order = [0, 2, 1, 3]
        
        if color_order.lower() == "gbrg":
            self.color_order = [1, 2, 0, 1]
        elif color_order.lower() == "rggb":
            self.color_order = [0, 1, 1, 2]
        elif color_order.lower() == "bggr":
            self.color_order = [2, 1, 1, 0]
            
        # top left (tl)
        self.mask[self.polar_order[0], self.color_order[0],  ::4,  ::4] = 1 #polar 90 R
        self.mask[self.polar_order[1], self.color_order[0],  ::4, 1::4] = 1 #polar 45 R
        self.mask[self.polar_order[2], self.color_order[0], 1::4,  ::4] = 1 #polar 135 R
        self.mask[self.polar_order[3], self.color_order[0], 1::4, 1::4] = 1 #polar 0 R
        # top right (tr)
        self.mask[self.polar_order[0], self.color_order[1],  ::4, 2::4] = 1 #polar 90 G
        self.mask[self.polar_order[1], self.color_order[1],  ::4, 3::4] = 1 #polar 45 G
        self.mask[self.polar_order[2], self.color_order[1], 1::4, 2::4] = 1 #polar 135 G
        self.mask[self.polar_order[3], self.color_order[1], 1::4, 3::4] = 1 #polar 0 G
        # bottom left (bl)
        self.mask[self.polar_order[0], self.color_order[2], 2::4,  ::4] = 1 #polar 90 G
        self.mask[self.polar_order[1], self.color_order[2], 2::4, 1::4] = 1 #polar 45 G
        self.mask[self.polar_order[2], self.color_order[2], 3::4,  ::4] = 1 #polar 135 G
        self.mask[self.polar_order[3], self.color_order[2], 3::4, 1::4] = 1 #polar 0 G
        # bottom right (br)
        self.mask[self.polar_order[0], self.color_order[3], 2::4, 2::4] = 1 #polar 90 B
        self.mask[self.polar_order[1], self.color_order[3], 2::4, 3::4] = 1 #polar 45 B
        self.mask[self.polar_order[2], self.color_order[3], 3::4, 2::4] = 1 #polar 135 B
        self.mask[self.polar_order[3], self.color_order[3], 3::4, 3::4] = 1 #polar 0 B

        self.polar_mask = torch.amax(self.mask, 1).to(dtype=torch.bool)
        self.color_mask = torch.amax(self.mask, 0)
        self.color_mask = self.color_mask[:,::2,::2].to(dtype=torch.bool)
        self.color_polar_mask = self.mask.view(12, max_height, max_width)

    def to(self, device):
        self.mask = self.mask.to(device=device)
        self.color_polar_mask = self.color_polar_mask.to(device=device)
        self.polar_mask = self.polar_mask.to(device=device)
        self.color_mask = self.color_mask.to(device=device)

    def generate_raw_CPFA(self, full_color_polar, pattern_sp=[0,0]):
        # full_color_polar: (4, 3, h, w)
        h, w = full_color_polar.shape[-2:]
        raw_CPFA = torch.unsqueeze(torch.sum(
            full_color_polar*
            self.mask[:,:,pattern_sp[0]:pattern_sp[0]+h,pattern_sp[1]:pattern_sp[1]+w],
            (0, 1)), 0)
        # raw_CPFA: (1, h, w)
        return raw_CPFA

    def generate_sub_bayer(self, raw_CPFA):
        # raw_CPFA: (batch, 1, h, w)
        sub_bayer = sub_sample(raw_CPFA) # (batch, 4, 1, h/2, w/2)
        return sub_bayer

    def generate_polar_mosaic(self, sub_color):
        # sub_color: (batch, 4, 3, h/2, w/2)
        sub_color = torch.transpose(sub_color, 1, 2) #(batch, 3, 4, h/2, w/2)
        polar_mosaic = nn.PixelShuffle(2)(sub_color) #(batch, 3, 1, h, w)
        return polar_mosaic

    def sparse_sub_bayer(self, sub_bayer):
        # sub_bayer: (batch, 1, h/2, w/2)
        return sub_image(sub_bayer, self.color_mask) #(batch, 3, h/2, w/2)

    def sparse_polar_mosaic(self, polar_mosaic):
        # polar_mosaic: (batch, 1, h, w)
        return sub_image(polar_mosaic, self.polar_mask) #(batch, 4, h, w)

    def sparse_color_polar_mosaic(self, raw_CPFA):
        return sub_image(raw_CPFA, self.color_polar_mask)

    def sub_color_image(self, full_color_polar):
        # full_color_polar: (4, 3, h, w)

        tl_idx, tr_idx, bl_idx, br_idx = self.polar_order
        
        tl = full_color_polar[tl_idx,:,::2,::2]
        tr = full_color_polar[tr_idx,:,::2,1::2]
        bl = full_color_polar[bl_idx,:,1::2,::2]
        br = full_color_polar[br_idx,:,1::2,1::2]

        y = torch.stack([tl, tr, bl, br], axis=0)

        return y
