from trainer import *
from utils import get_config, get_all_data_loaders, prepare_sub_folder, get_CPFA, get_model_name_from_config_path
from CPFANetBase import CPFANetBase
import argparse
from pathlib import Path
from CPFA import CPFA
import os
import shutil
import torch
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse argument

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml', help='Path to the config file.')
parser.add_argument('--output', type=str, default='outputs', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

# Load config

config = get_config(opts.config)

# Load data

train_loader, trainv_loader, val_loader, test_loader = get_all_data_loaders(config)


# prepare log and save dirs
model_name = get_model_name_from_config_path(opts.config)
if not opts.resume and os.path.isdir(os.path.join(opts.output, model_name)):
    ans = None
    while ans not in ['y', 'n']:
        ans = input('Do you want to resume the model? (y/n)\t')
    if ans == 'y':
       opts.resume = True 
ckp_dir, log_dir = prepare_sub_folder(opts.output, opts.resume, model_name)
shutil.copy(opts.config, os.path.join(Path(ckp_dir).parent.absolute(), 'config.yaml')) # copy config file to output folder

# load model

cpfa = get_CPFA(config)
cpfa.to(device=device)
net = CPFANetBase.build_net(config, cpfa)
trainer = globals()[config['opt_approach']](net, ckp_dir, config, log_dir, val_loader, trainv_loader, test_loader)
trainer.to(device=device)
if opts.resume:
    print("Resuming the network...")
    trainer.resume(config)
    print("Done loading!")

if TwoStepTrainer in trainer.__class__.__bases__:
    while True:
        for raw_CPFA, gt, sub_color_gt in train_loader:

            # merge crops with batches
            raw_CPFA = raw_CPFA.view(-1, raw_CPFA.shape[2], raw_CPFA.shape[3], raw_CPFA.shape[4]).to(device=device)
            sub_color_gt = sub_color_gt.view(-1, sub_color_gt.shape[2], sub_color_gt.shape[3], sub_color_gt.shape[4], sub_color_gt.shape[5]).to(device=device)
            gt = gt.view(-1, gt.shape[2], gt.shape[3], gt.shape[4], gt.shape[5]).to(device=device)
            # training
            trainer.update(raw_CPFA, sub_color_gt, gt)
            current_epoch = trainer.step/trainer.steps_per_epoch
            if 'max_epoch' in config and current_epoch >= config['max_epoch']:
                sys.exit()
            
elif trainer.__class__ in [OneStepTrainer, OneStepMultiDecoderTrainer]:
    while True:
        for raw_CPFA, gt, _ in train_loader:

            # merge crops with batches
            raw_CPFA = raw_CPFA.view(-1, raw_CPFA.shape[2], raw_CPFA.shape[3], raw_CPFA.shape[4]).to(device=device)
            gt = gt.view(-1, gt.shape[2], gt.shape[3], gt.shape[4], gt.shape[5]).to(device=device)
            # training
            full_color_polar = net(raw_CPFA)
            trainer.update(full_color_polar, gt)
            current_epoch = trainer.step/trainer.steps_per_epoch
            if 'max_epoch' in config and current_epoch >= config['max_epoch']:
                sys.exit()

