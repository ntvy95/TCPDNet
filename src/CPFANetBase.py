import torch.nn as nn
from net_utils import bilinear_interp
import torch
from RefineNet import RefineNet

class CPFANetBase(nn.Module):
    def __init__(self, CPFA):
        super(CPFANetBase, self).__init__()
        self.CPFA = CPFA
    
    @staticmethod
    def build_net(config, cpfa):
        net_approach = config['net_approach']
        assert CPFANetBase.is_valid_approach(net_approach)
        color_net_method = config['color_net']['method'] if 'color_net' in config else None
        polar_net_method = config['polar_net']['method'] if 'polar_net' in config else None
        if color_net_method and polar_net_method:
            net = globals()[net_approach](cpfa, color_net_method, polar_net_method, config)
        else:
            net_method = config['net']['method']
            net = globals()[net_approach](cpfa, net_method, config)
        return net

    @staticmethod
    def is_valid_approach(net_approach):
        return net_approach in ['OneStepColorPolarNet', 'TwoStepColorPolarNet']

class OneStepColorPolarNet(CPFANetBase):
    def __init__(self, CPFA, net, cfg):
        super(OneStepColorPolarNet, self).__init__(CPFA)
        assert self.is_valid_method(net)
        self.net = globals()[net](self.CPFA, cfg['net'])

    def forward(self, raw_CPFA):
        full_color_polar = self.net(raw_CPFA)
        h = full_color_polar.shape[-2]
        w = full_color_polar.shape[-1]
        full_color_polar = full_color_polar.view(-1, 4, 3, h, w)
        return full_color_polar

    @staticmethod
    def is_valid_method(net):
        return net in ['OneStepBilinear', 'OneStepBilinearRefine']

class OneStepNet(CPFANetBase):
    def __init__(self, CPFA, cfg = None):
        super(OneStepNet, self).__init__(CPFA)

class OneStepBilinear(OneStepNet):
    def forward(self, raw_CPFA):
        sparse_color_polar_mosaic = self.CPFA.sparse_color_polar_mosaic(raw_CPFA)
        full_color_polar = torch.zeros_like(sparse_color_polar_mosaic)
        full_color_polar[:,1::3,:,:] = bilinear_interp(sparse_color_polar_mosaic[:,1::3,:,:], is_green=True, freq=16)
        full_color_polar[:,::3,:,:] = bilinear_interp(sparse_color_polar_mosaic[:,::3,:,:],freq=16)
        full_color_polar[:,2::3,:,:] = bilinear_interp(sparse_color_polar_mosaic[:,2::3,:,:],freq=16)
        return full_color_polar

class OneStepBilinearRefine(OneStepNet):
    def __init__(self, CPFA, cfg = None):
        super(OneStepBilinearRefine, self).__init__(CPFA)
        print("OneStepBilinearRefine")
        
        self.bilinear = OneStepBilinear(self.CPFA)
        net_setting = cfg['net_setting'] if 'net_setting' in cfg else None
        self.refine_net = RefineNet(cfg['net_base'], 12, 12,
                                    cfg['skip_connection_flag'],
                                    net_setting)

    def forward(self, raw_CPFA):
        full_color_polar = self.bilinear(raw_CPFA)
        full_color_polar = self.refine_net(full_color_polar)
        return full_color_polar


class TwoStepColorPolarNet(CPFANetBase):
    def __init__(self, CPFA, color_net, polar_net, cfg=None):
        super(TwoStepColorPolarNet, self).__init__(CPFA)
        print("TwoStepColorPolarNet")
        
        if cfg:
            color_net_cfg = cfg['color_net']
            polar_net_cfg = cfg['polar_net']
        else:
            color_net_cfg = None
            polar_net_cfg = None
            self.original_pixel_replacement = False

        assert self.is_valid_color_net(color_net)
        assert self.is_valid_polar_net(polar_net)
        self.color_net = globals()[color_net](self.CPFA, color_net_cfg)
        self.polar_net = globals()[polar_net](self.CPFA, polar_net_cfg)

    @staticmethod
    def is_valid_color_net(color_net):
        return color_net in ['ColorBilinear', 'ColorBilinearRefine']

    @staticmethod
    def is_valid_polar_net(polar_net):
        return polar_net in ['PolarBilinear', 'PolarBilinearRefine', 'PolarBilinearDisentangle']

    def sub_colors(self, sub_bayers):
        sub_colors = []
        for i in range(4):
            sub_color = self.color_net(sub_bayers[:,i,:,:,:])
            sub_colors.append(sub_color)
        sub_colors = torch.stack(sub_colors, axis=1)
        return sub_colors

    def full_color_polar(self, sub_colors):
        full_color_polar = []
        polar_mosaic = self.CPFA.generate_polar_mosaic(sub_colors)
        for i in range(3):
            full_polar = self.polar_net(polar_mosaic[:,i,:,:,:])
            full_color_polar.append(full_polar)
        full_color_polar = torch.stack(full_color_polar, axis=2)
        return full_color_polar

    def forward(self, raw_CPFA, *, polar_net = True):
        sub_bayers = self.CPFA.generate_sub_bayer(raw_CPFA)
        sub_colors = self.sub_colors(sub_bayers)
        full_color_polar = self.full_color_polar(sub_colors) if polar_net else []
        return sub_colors, full_color_polar    

class ColorNet(CPFANetBase):
    def __init__(self, CPFA, cfg=None):
        super(ColorNet, self).__init__(CPFA)

class PolarNet(CPFANetBase):
    def __init__(self, CPFA, cfg=None):
        super(PolarNet, self).__init__(CPFA)

class ColorBilinear(ColorNet):
    def forward(self, sub_bayer):
        sparse_sub_bayer = self.CPFA.sparse_sub_bayer(sub_bayer)
        sub_color = torch.zeros_like(sparse_sub_bayer)
        sub_color[:,[0,2],:,:] = bilinear_interp(sparse_sub_bayer[:,[0,2],:,:])
        sub_color[:,1,:,:] = bilinear_interp(sparse_sub_bayer[:,1,:,:].unsqueeze(1), is_green=True).squeeze(1)
        return sub_color

class PolarBilinear(PolarNet):
    def forward(self, polar_mosaic):
        sparse_polar_mosaic = self.CPFA.sparse_polar_mosaic(polar_mosaic)
        full_polar = bilinear_interp(sparse_polar_mosaic)
        return full_polar

class ColorBilinearRefine(ColorNet):
    def __init__(self, CPFA, cfg):
        super(ColorBilinearRefine, self).__init__(CPFA)
        print("ColorBilinearRefine")
        
        self.bilinear = ColorBilinear(self.CPFA)
        net_setting = cfg['net_setting'] if 'net_setting' in cfg else None
        self.refine_net = RefineNet(cfg['net_base'], 3, 3,
                                    cfg['skip_connection_flag'],
                                    net_setting)

    def forward(self, sub_bayer):
        sub_color = self.bilinear(sub_bayer)
        sub_color = self.refine_net(sub_color)
        return sub_color

class PolarBilinearRefine(PolarNet):
    def __init__(self, CPFA, cfg):
        super(PolarBilinearRefine, self).__init__(CPFA)
        print("PolarBilinearRefine")
        
        self.bilinear = PolarBilinear(self.CPFA)
        net_setting = cfg['net_setting'] if 'net_setting' in cfg else None
        self.refine_net = RefineNet(cfg['net_base'], 4, 4,
                                    cfg['skip_connection_flag'],
                                    net_setting)
        
    def forward(self, polar_mosaic):
        full_polar = self.bilinear(polar_mosaic)
        full_polar = self.refine_net(full_polar)
        return full_polar
