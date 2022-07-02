from dataset import BaseTestingDataset, PatchDataset
import torch
from metrics import eval_all
import os
import time
import numpy as np
from utils import get_config, get_all_data_loaders, get_CPFA, get_model_list, get_model_name_from_config_path
from CPFANetBase import CPFANetBase
from torch.utils.data import DataLoader
from loss import LossL1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_net(net, loader, writer = None, tag = None, epoch = None, pattern_sp = None):

    net.eval()

    avg_metrics = {}

    l1_loss = 0
    l1_loss_est = LossL1(torch.mean, axis = 1)
    
    torch.backends.cudnn.deterministic = True

    for idx, raw_CPFA, gt, _ in loader:
        raw_CPFA = raw_CPFA.to(device=device)
        gt = gt.to(device=device)
        
        result = net(raw_CPFA)
        if type(result) is tuple:
            _, y = result
        else:
            y = result

        metrics = eval_all(y, gt)
        for metric, value in metrics.items():
            if metric not in avg_metrics:
                avg_metrics[metric] = 0
            avg_metrics[metric] += value.item()*len(raw_CPFA)
            
        l1_loss += l1_loss_est(y, gt).mean()*len(raw_CPFA)

    N = len(loader.dataset)
    
    for metric, value in avg_metrics.items():
        avg_metrics[metric] /= N
        print(f'{metric}: {avg_metrics[metric]}')
        
    l1_loss = l1_loss/N
    print(f'L1 loss: {l1_loss}')

    if writer:
        for metric, value in avg_metrics.items():
            writer.add_scalar(tag + metric, value, epoch)
        writer.add_scalar(tag + 'Error/L1 loss',  l1_loss.item(), epoch)
        net.train()
        return avg_metrics['CPSNR/Average']
    else:
        return avg_metrics

from trainer import *
import argparse

def load_model(cfg_path):
    config = get_config(cfg_path)
    train_loader, trainv_loader, val_loader, test_loader = get_all_data_loaders(config)
    cpfa = get_CPFA(config)
    cpfa.to(device=device)
    net = CPFANetBase.build_net(config, cpfa).to(device=device)
    return net, test_loader,  config

def best_eval(cfg_path):
    net, test_loader,  config = load_model(cfg_path)
    if 'opt_approach' in config:
        output_directory = os.path.join("outputs", model_name)
        ckp_dir = os.path.join(output_directory, "checkpoints")
        trainer = globals()[config['opt_approach']](net, ckp_dir, config)
        trainer.load_best(config, True)
        trainer.to(device=device)
        trainer.eval()
    else:
        net.eval()

    with torch.no_grad():
        eval_net(net, test_loader)

def avg_5_models(cfg_path):
    net, test_loader, config = load_model(cfg_path)
    model_name = get_model_name_from_config_path(cfg_path)
    output_directory = os.path.join("outputs", model_name)
    if 'opt_approach' not in config:
        raise ValueError('avg_5_models is not applicable to non-trainable methods.')
    metrics_list = {}
    for i in range(1,6):
        tail = "" if i == 1 else "_T_" + str(i)
        ckp_dir = os.path.join(output_directory + tail, "checkpoints")
        if not os.path.isdir(ckp_dir):
            raise OSError(f'Please manually rename the checkpoint directories of 4 trained models to the format: <model_name>_T_<a number in range [2-5]>. At least one trained model should have the checkpoint directory name remains intact. Currently, the error is raised because This directory does not exist: {ckp_dir}')
        checkpoint = get_model_list(ckp_dir, "net")[-1]
        trainer = globals()[config['opt_approach']](net, ckp_dir, config)
        trainer.resume(config, checkpoint, not torch.cuda.is_available())
        trainer.to(device=device)
        trainer.eval()
        with torch.no_grad():
            metrics = eval_net(net, test_loader)
            for metric, value in metrics.items():
                if metric not in metrics_list:
                    metrics_list[metric] = []
                metrics_list[metric].append(value)
    mean_std(metrics_list)


def mean_std(metrics_list):
    for metric in metrics_list:
        mean = np.mean(metrics_list[metric])
        std = np.std(metrics_list[metric])
        if 'RMSE' in metric:
            print(f'{metric}: {round(mean, 3)} ± {round(std, 3)}')
        else:
            print(f'{metric}: {round(mean, 2)} ± {round(std, 2)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='unet_sim', help='Path to the config file.')
    parser.add_argument('--mode', type=str, default='free_eval', help='Evaluation type.')
    opts = parser.parse_args()
    if opts.mode == 'best_eval':
        best_eval(opts.config)
    elif opts.mode == 'avg_5_models':
        avg_5_models(opts.config)


