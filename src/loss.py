# Credit to: mtanaka@sc.e.titech.ac.jp

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_stokes, compute_dop, rgb_to_ycbr


class LossCompose(nn.Module):
    def __init__(self): # losses : [ (loss, alpha) ]
        super(LossCompose, self).__init__()
        self.losses = []

    def append( self, loss, alpha, name ):
        if( alpha > 0 ):
            self.losses.append( (loss, alpha, name) )

    def forward( self, sRGB, gt_sRGB ):
        loss = 0
        loss_track = {}
        for (_loss, _alpha, _name) in self.losses:
            temp = _loss( sRGB, gt_sRGB ).mean()
            loss = loss + temp * _alpha
            loss_track[_name] = temp.item()
        return loss, loss_track

class LossL1(nn.Module):
    def __init__(self, op, axis):
        super(LossL1, self).__init__()
        self.op = op
        self.axis = axis

    def forward( self, sRGB, gt_sRGB ):
        loss = self.op(torch.abs( sRGB - gt_sRGB ), axis=self.axis).mean( (-1,-2) )
        return loss

class LossL2(nn.Module):
    def __init__(self, op, axis):
        super(LossL2, self).__init__()
        self.op = op
        self.axis = axis

    def forward( self, sRGB, gt_sRGB ):
        loss = self.op(torch.square( sRGB - gt_sRGB ), axis=self.axis).mean( (-1,-2) )
        return loss

class LossLp(nn.Module):
    def __init__(self, p):
        super(LossLp, self).__init__()
        self.p = p

    def forward( self, sRGB, gt_sRGB ):
        loss = torch.abs( sRGB - gt_sRGB ).pow(self.p).sum(1).mean( (1,2) )
        return loss

class LossGrad(nn.Module):
    def __init__(self, loss):
        super(LossGrad, self).__init__()
        self.loss = loss

    def forward( self, sRGB, gt_sRGB ):
        sRGBh = sRGB[:,:,:-1,:] - sRGB[:,:,1:,:]
        sRGBv = sRGB[:,:,:,:-1] - sRGB[:,:,:,1:]
        gt_sRGBh = gt_sRGB[:,:,:-1,:] - gt_sRGB[:,:,1:,:]
        gt_sRGBv = gt_sRGB[:,:,:,:-1] - gt_sRGB[:,:,:,1:]
        return self.loss( sRGBh, gt_sRGBh ) + self.loss( sRGBv, gt_sRGBv )

class RMSE255(nn.Module):
    def __init__(self):
        super(RMSE255, self).__init__()

    def forward( self, sRGB, gt_sRGB ):
        return ( sRGB - gt_sRGB ).square().sum( 1 ).mean( (1,2) ).sqrt() * 255

class CPSNR(nn.Module):
    def __init__(self):
        super(CPSNR, self).__init__()

    def forward( self, sRGB, gt_sRGB ):
        mse = ( sRGB - gt_sRGB ).square().sum( 1 ).mean( (1,2) )
        return -10*torch.log10( mse )

class LossYCbCr(nn.Module):
    def __init__(self, dist):
        super(LossYCbCr, self).__init__()
        self.dist = dist(torch.mean, 1)

    def forward(self, pred, gt):
        pred = rgb_to_ycbr(pred)
        gt = rgb_to_ycbr(gt)
        loss = self.dist(pred, gt).mean(1)
        return loss    

class BaseLoss(nn.Module):
    def __init__(self, loss_weights):
        super(BaseLoss, self).__init__()

        self.loss = LossCompose()

        if 'l1_w' in loss_weights and loss_weights['l1_w'] > 0:
            self.loss.append(LossL1(torch.mean, axis=1), loss_weights['l1_w'], 'l1')
            
    def forward(self, pred, gt):
        h, w = pred.shape[-2:]
        h = int(h)
        w = int(w)
        pred = pred.view(-1, 12, h, w)
        gt = gt.view(-1, 12, h, w)
        return self.loss(pred, gt)

class ColorLoss(BaseLoss):
    pass

class PolarLoss(BaseLoss):

    def __init__(self, loss_weights):
        super(PolarLoss, self).__init__(loss_weights)

        self.polar_loss = LossCompose()
            
        if 'ycbrl1_w' in loss_weights and loss_weights['ycbrl1_w'] > 0:
            self.polar_loss.append(LossYCbCr(LossL1), loss_weights['ycbrl1_w'], 'ycbrl1')

            
    def forward(self, pred, gt):
        ploss, ploss_track = self.polar_loss(pred, gt)
        loss, loss_track = super(PolarLoss, self).forward(pred, gt)
        loss_track.update(ploss_track)
        return ploss + loss, loss_track

class ColorPolarLoss(BaseLoss):
    pass

if( __name__ == '__main__' ):
    loss = LossCompose()
    loss.append( LossL1(torch.sum, 1), 1 )
    loss.append( LossGrad( LossL1(torch.sum, 1) ), 2 )
    # loss = |x-y|+2 |nabla(x)-nabla(y)|
