from dataset import BaseTestingDataset
from trainer import TwoStepSimultaneousTrainer, OneStepTrainer
from utils import get_config, get_all_data_loaders, get_CPFA, get_model_list, compute_stokes, compute_aop, compute_dop
from matplotlib.colors import hsv_to_rgb
from torch.utils.data import DataLoader
import PIL.Image as Image
import numpy as np
import os
import torch
from CPFANetBase import CPFANetBase

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg_path = 'configs/unet_sim_OL_ycbr_4.yaml'
config = get_config(cfg_path)
train_loader, trainv_loader, val_loader, test_loader = get_all_data_loaders(config)
cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
output_directory = os.path.join("outputs", cfg_name)
ckp_dir = os.path.join(output_directory, "checkpoints")
cpfa = get_CPFA(config)
cpfa.to(device=device)
net = CPFANetBase.build_net(config, cpfa).to(device=device)
if 'opt_approach' in config:
    trainer = globals()[config['opt_approach']](net, ckp_dir, config)
    trainer.to(device=device)
    trainer.load_best(config, is_cpu=True)
    trainer.eval()
folder = os.path.join('visualization', cfg_name)
if not os.path.exists(folder):
    os.makedirs(folder)

data_loader = test_loader

def tensor2img(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.numpy()
    return Image.fromarray((tensor*255).astype(np.uint8))

with torch.no_grad():
    for idx, raw_CPFA, gt, sub_color_gt in data_loader:
        file_name = data_loader.dataset.flist[idx]
        file_name = file_name.replace('.hdf5', '')

        if not os.path.exists(os.path.join(folder, file_name)):
            os.makedirs(os.path.join(folder, file_name))
        
        raw_CPFA = raw_CPFA.to(device=device)

        result = net(raw_CPFA)
        if type(result) is tuple:
            sub_colors, full_color_polar = result
        else:
            full_color_polar = result
        gt_s, _ = compute_stokes(gt)
        gt_dop = compute_dop(gt_s)
        gt_aop = compute_aop(gt_s)
        gt_dop = gt_dop.cpu().detach()
        gt_aop = gt_aop.cpu().detach()
        full_color_polar_s, _ = compute_stokes(full_color_polar)
        full_color_polar_dop = compute_dop(full_color_polar_s)
        full_color_polar_aop = compute_aop(full_color_polar_s)
        full_color_polar = full_color_polar.cpu().detach()
        full_color_polar_dop = full_color_polar_dop.cpu().detach()
        full_color_polar_aop = full_color_polar_aop.cpu().detach()
        full_color_polar = np.transpose(full_color_polar[0], [0,2,3,1])
        full_color_polar[full_color_polar < 0] = 0
        full_color_polar[full_color_polar > 1] = 1
        gt = np.transpose(gt[0], [0,2,3,1])
        gt[gt < 0] = 0
        gt[gt > 1] = 1
        polar_idx = [0, 45, 90, 135]
        color_idx = ['R', 'G', 'B']
        for i in range(4):
            diff = torch.mean(1 - torch.abs(full_color_polar[i] - gt[i]), axis=2).unsqueeze(-1).repeat(1,1,3)
            img_diff, img_gt, img_est = tensor2img(diff), tensor2img(gt[i]), tensor2img(full_color_polar[i])
            img_gt.save(os.path.join(folder, file_name, f"{polar_idx[i]}.png"))
            img_est.save(os.path.join(folder, file_name, f"{polar_idx[i]}_{file_name}.png"))
            img_diff.save(os.path.join(folder, file_name, f"{polar_idx[i]}_{file_name}_diff.png"))
        for i in range(3):
            img_gt, img_est = tensor2img(1 - gt_dop[0][i]), tensor2img(1 - full_color_polar_dop[0][i])
            img_gt.save(os.path.join(folder, file_name, f"{color_idx[i]}_DoP.png"))
            img_est.save(os.path.join(folder, file_name, f"{color_idx[i]}_DoP_{file_name}.png"))
        h, w = gt.shape[-3:-1]
        for i in range(3):
            gt_aop_hsv = hsv_to_rgb(np.stack([gt_aop[0][i].flatten(), np.ones(h*w), np.ones(h*w)], axis=1)).reshape(h, w, 3)
            full_color_polar_aop_hsv = hsv_to_rgb(np.stack([full_color_polar_aop[0][i].flatten(), np.ones(h*w), np.ones(h*w)], axis=1)).reshape(h, w, 3)
            img_gt, img_est = tensor2img(gt_aop_hsv), tensor2img(full_color_polar_aop_hsv)
            img_gt.save(os.path.join(folder, file_name, f"{color_idx[i]}_AoP.png"))
            img_est.save(os.path.join(folder, file_name, f"{color_idx[i]}_AoP_{file_name}.png"))
        for i in range(3):
            gt_aop_dop_hsv = hsv_to_rgb(np.stack([gt_aop[0][i], torch.sqrt(gt_dop[0][i]), np.ones((h, w))], axis=2))
            full_color_polar_aop_dop_hsv = hsv_to_rgb(np.stack([full_color_polar_aop[0][i], torch.sqrt(full_color_polar_dop[0][i]), np.ones((h, w))], axis=2))
            img_gt, img_est = tensor2img(gt_aop_dop_hsv), tensor2img(full_color_polar_aop_dop_hsv)
            img_gt.save(os.path.join(folder, file_name, f"{color_idx[i]}_AoP_DoP.png"))
            img_est.save(os.path.join(folder, file_name, f"{color_idx[i]}_AoP_DoP_{file_name}.png"))


