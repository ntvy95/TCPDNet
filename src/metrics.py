import torch.nn as nn
import torch
from utils import compute_stokes, compute_aop, compute_dop

'''
CPSNR is a Pytorch version of:
https://github.com/ymonno/EARI-Polarization-Demosaicking/blob/master/Functions/TIP_RI/imcpsnr.m
'''

def CPSNR(x, y, peak=1, b=15):
    # dim info of x: (batch, rgb, h, w)
    # dim info of y: (batch, rgb, h, w)
    # b: a scalar
    x = x[:, :, b-1:x.shape[2]-b, b-1:x.shape[3]-b]
    y = y[:, :, b-1:y.shape[2]-b, b-1:y.shape[3]-b]
    mse = nn.MSELoss(reduction="none")
    batch_psnr = mse(x, y).detach()
    cpsnr = torch.mean(10*torch.log10(peak**2/(torch.mean(batch_psnr, (1, 2, 3)) + 1e-32)))
    return cpsnr

'''
angle_error is a Pytorch version of:
https://github.com/ymonno/EARI-Polarization-Demosaicking/blob/master/Functions/angleerror_AOLP.m
'''

def angle_error(x, y, b=15):
    # dim info of x: (batch, rgb/polar, h, w)
    # dim info of y: (batch, rgb/polar, h, w)
    # b: a scalar
    x = x[:, :, b-1:x.shape[2]-b, b-1:x.shape[3]-b]
    y = y[:, :, b-1:y.shape[2]-b, b-1:y.shape[3]-b]
    mse = nn.MSELoss(reduction="none")
    batch_error,_ = torch.min(torch.stack([mse(x, y).detach(), mse(x-1,y).detach(), mse(x+1,y).detach()], axis=0), 0)
    angle_error = torch.mean(torch.sqrt(torch.mean(batch_error, (1, 2, 3)))*180)
    return angle_error

def RMSE(x, y, b=15):
    x = x[:, :, b-1:x.shape[2]-b, b-1:x.shape[3]-b]
    y = y[:, :, b-1:y.shape[2]-b, b-1:y.shape[3]-b]
    mse = nn.MSELoss(reduction='none')
    rmse = torch.mean(torch.sqrt(torch.mean(mse(x*255, y*255), (1,2,3))))
    return rmse
    

def eval_all(y, gt):
    y_s, y_scaled_s = compute_stokes(y)
    gt_s, gt_scaled_s = compute_stokes(gt)

    s0_cpsnr = CPSNR(y_scaled_s[:,0,:,:,:], gt_scaled_s[:,0,:,:,:])
    s1_cpsnr = CPSNR(y_scaled_s[:,1,:,:,:], gt_scaled_s[:,1,:,:,:])
    s2_cpsnr = CPSNR(y_scaled_s[:,2,:,:,:], gt_scaled_s[:,2,:,:,:])

    i0_cpsnr = CPSNR(y[:,0,:,:,:], gt[:,0,:,:,:])
    i45_cpsnr = CPSNR(y[:,1,:,:,:], gt[:,1,:,:,:])
    i90_cpsnr = CPSNR(y[:,2,:,:,:], gt[:,2,:,:,:])
    i135_cpsnr = CPSNR(y[:,3,:,:,:], gt[:,3,:,:,:])

    y_dop = compute_dop(y_s)
    gt_dop = compute_dop(gt_s)
    dop_cpsnr = CPSNR(y_dop, gt_dop)
    
    y_aop = compute_aop(y_s)
    gt_aop = compute_aop(gt_s)
    aop_angle_error = angle_error(y_aop, gt_aop)

    s0_rmse = RMSE(y_scaled_s[:,0,:,:,:], gt_scaled_s[:,0,:,:,:])
    s1_rmse = RMSE(y_scaled_s[:,1,:,:,:], gt_scaled_s[:,1,:,:,:])
    s2_rmse = RMSE(y_scaled_s[:,2,:,:,:], gt_scaled_s[:,2,:,:,:])

    i0_rmse = RMSE(y[:,0,:,:,:], gt[:,0,:,:,:])
    i45_rmse = RMSE(y[:,1,:,:,:], gt[:,1,:,:,:])
    i90_rmse = RMSE(y[:,2,:,:,:], gt[:,2,:,:,:])
    i135_rmse = RMSE(y[:,3,:,:,:], gt[:,3,:,:,:])
    
    dop_rmse = RMSE(y_dop, gt_dop)

    avg_cpsnr = (i0_cpsnr + i45_cpsnr + i90_cpsnr + i135_cpsnr + dop_cpsnr + s0_cpsnr + s1_cpsnr + s2_cpsnr)/8
    
    return { 'CPSNR/I0': i0_cpsnr,
             'CPSNR/I45': i45_cpsnr,
             'CPSNR/I90': i90_cpsnr,
             'CPSNR/I135': i135_cpsnr,
             'CPSNR/S0': s0_cpsnr,
             'CPSNR/S1': s1_cpsnr,
             'CPSNR/S2': s2_cpsnr,
             'CPSNR/DoP': dop_cpsnr,
             'CPSNR/Average': avg_cpsnr,
             'Error/AoP (angle error)': aop_angle_error }
