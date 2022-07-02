# A modification of https://github.com/NVlabs/MUNIT/blob/master/trainer.py

import torch.nn as nn
from utils import get_model_list, get_optimizer, get_scheduler, get_config, weights_init, get_epoch_performance
from CPFANetBase import *
from loss import ColorLoss, PolarLoss, ColorPolarLoss
from torch.optim import lr_scheduler
from evals import eval_net
import os
from logger import Writer
from metrics import eval_all

class Trainer(nn.Module):
    def __init__(self, net, ckp_dir, hyperparameters, log_dir = None, val_loader = None, trainv_loader = None, test_loader = None):
        super(Trainer, self).__init__()
        self.ckp_dir = ckp_dir
        self.log_dir = log_dir
        if self.log_dir:
            self.writer = Writer(self.log_dir)
        self.val_loader = val_loader
        self.trainv_loader = trainv_loader
        self.test_loader = test_loader
        self.net = net
        self.step = 0
        self.steps_per_training_log = hyperparameters['steps_per_training_log']
        self.steps_per_epoch = hyperparameters['steps_per_epoch']

    def write_loss(self, tag, track, write_now=True, avg_per_epoch=False):
        for k, v in track.items():
            self.writer.add_scalar(tag + k, v, self.step, write_now, avg_per_epoch)

class OneStepTrainer(Trainer):
    def __init__(self, net, ckp_dir, hyperparameters, log_dir = None, val_loader = None, trainv_loader = None, test_loader = None):
        super(OneStepTrainer, self).__init__(net, ckp_dir, hyperparameters, log_dir, val_loader, trainv_loader, test_loader)
        net_params = list(self.net.parameters())
        self.net_opt = get_optimizer(hyperparameters['net_opt'], net_params)
        self.net_scheduler = get_scheduler(self.net_opt, hyperparameters['net_opt']['lr'])
        self.net.apply(weights_init(hyperparameters['net']['init']))
        if hasattr(self.net.net.refine_net.net, 'outc'):
            self.net.net.refine_net.net.outc.apply(weights_init('zero'))
        self.net_loss = PolarLoss(hyperparameters['net_loss'])

    def compute_loss(self, full_color_polar, gt):
        net_loss, net_loss_track = self.net_loss(full_color_polar, gt)
        return net_loss, net_loss_track

    def update_after(self, full_color_polar, gt, total_loss, net_loss_track):
        self.step += 1
        if self.steps_per_training_log and self.step % self.steps_per_training_log == 0:
            self.writer.add_scalar('Train/Total', total_loss, self.step, avg_per_epoch=True)
            self.write_loss('Train/', net_loss_track, avg_per_epoch=True)
            metrics = eval_all(full_color_polar, gt)
            for metric, value in metrics.items():
                metrics[metric] = value.mean().item()
                self.writer.add_scalar(f'Train/{metric}', metrics[metric], self.step, write_now = False, avg_per_epoch = True)
            
        if self.step % self.steps_per_epoch == 0:
            with torch.no_grad():
                print('Epoch average metrics:')
                self.writer.write_avg_epoch_metrics(self.step/self.steps_per_epoch)
                print('Validation set:')
                avg_val = eval_net(self.net, self.val_loader, self.writer, 'Val/', self.step/self.steps_per_epoch)
                self.update_learning_rate(avg_val)
                print('Test set:')
                eval_net(self.net, self.test_loader, self.writer, 'Test/', self.step/self.steps_per_epoch)
            self.save(avg_val)
            print("Done saving the checkpoint of epoch " + str(int(self.step/self.steps_per_epoch)) + "!")
            print(self.ckp_dir)

    def update_learning_rate(self, val_metrics = None):
        if self.net_scheduler:
            if val_metrics and type(self.net_scheduler) == lr_scheduler.ReduceLROnPlateau:
                self.net_scheduler.step(val_metrics)
            else:
                self.net_scheduler.step()
                
    def save(self, performance):
        opt_name = os.path.join(self.ckp_dir, 'optimizer.pt')
        current_epoch = self.step/self.steps_per_epoch
        net_name = os.path.join(self.ckp_dir, 'net_%08d.pt' % (current_epoch))
        torch.save(self.net.state_dict(), net_name)
        torch.save(self.net_opt.state_dict(), opt_name)
        final_checkpoint_file = os.path.join(self.ckp_dir, "final_checkpoint.txt")
        previous_checkpoint = os.path.join(self.ckp_dir, 'net_%08d.pt' % (current_epoch - 10))
        best_checkpoint = None
        best_performance = -float('inf')
        if os.path.isfile(final_checkpoint_file):
            with open(final_checkpoint_file) as f:
                best_checkpoint = f.read()
                best_epoch = int(best_checkpoint[-11:-3])
                best_performance = get_epoch_performance(self.log_dir, best_epoch)
            if current_epoch > 10 and best_checkpoint != previous_checkpoint:
                os.remove(previous_checkpoint)
        if performance > best_performance:
            if best_checkpoint is not None:
                best_epoch = int(best_checkpoint[-11:-3])
                if current_epoch - best_epoch >= 10:
                    os.remove(best_checkpoint)
            with open(final_checkpoint_file, 'w') as f:
                f.write(net_name)

    def load_best(self, hyperparameters, is_cpu=False):
        final_checkpoint_file = os.path.join(self.ckp_dir, "final_checkpoint.txt")
        with open(final_checkpoint_file) as f:
            best_checkpoint = f.read()
            self.resume(hyperparameters, best_checkpoint, is_cpu)

    def resume(self, hyperparameters, model_path=None, is_cpu=False):
        if not model_path:
            model_path = get_model_list(self.ckp_dir, "net")[-1] #get last model
        print(model_path)
        if is_cpu:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(model_path)
        epoch = int(model_path[-11:-3])
        self.step = epoch*self.steps_per_epoch
        self.net.load_state_dict(state_dict)
        if os.path.isfile(os.path.join(self.ckp_dir, 'optimizer.pt')):
            if is_cpu:
                state_dict = torch.load(os.path.join(self.ckp_dir, 'optimizer.pt'), map_location=torch.device('cpu'))
            else:
                state_dict = torch.load(os.path.join(self.ckp_dir, 'optimizer.pt'))
            self.net_opt.load_state_dict(state_dict)
            self.net_scheduler = get_scheduler(self.net_opt, hyperparameters['net_opt']['lr'], epoch)

    def update(self, full_color_polar, gt):
        net_loss, net_loss_track = self.compute_loss(full_color_polar, gt)
        self.net_opt.zero_grad()
        net_loss.backward()
        self.net_opt.step()
        self.update_after(full_color_polar, gt, net_loss.item(), net_loss_track)
        

class TwoStepTrainer(Trainer):
    def __init__(self, net, ckp_dir, hyperparameters, log_dir = None, val_loader = None, trainv_loader = None, test_loader = None):
        super(TwoStepTrainer, self).__init__(net, ckp_dir, hyperparameters, log_dir, val_loader, trainv_loader, test_loader)
        color_params = list(self.net.color_net.parameters())
        polar_params = list(self.net.polar_net.parameters())
        self.color_opt = get_optimizer(hyperparameters['color_opt'], color_params)
        self.polar_opt = get_optimizer(hyperparameters['polar_opt'], polar_params)
        self.color_scheduler = get_scheduler(self.color_opt, hyperparameters['color_opt']['lr'])
        self.polar_scheduler = get_scheduler(self.polar_opt, hyperparameters['polar_opt']['lr'])
        self.net.color_net.apply(weights_init(hyperparameters['color_net']['init']))
        self.net.color_net.refine_net.net.outc.apply(weights_init('zero'))
        self.net.polar_net.apply(weights_init(hyperparameters['polar_net']['init']))
        if hasattr(self.net.polar_net, 'refine_net'):
            self.net.polar_net.refine_net.net.outc.apply(weights_init('zero'))
        self.color_loss = ColorLoss(hyperparameters['color_loss'])
        self.polar_loss = PolarLoss(hyperparameters['polar_loss'])

    def compute_loss(self, sub_color, sub_color_gt, full_color_polar, gt):
        color_loss, color_loss_track = self.color_loss(sub_color, sub_color_gt)
        polar_loss, polar_loss_track = self.polar_loss(full_color_polar, gt)
        return color_loss, color_loss_track, polar_loss, polar_loss_track

    def update_after(self, total_loss, color_loss_track, polar_loss_track, full_color_polar, gt):
        self.step += 1
        if self.steps_per_training_log and self.step % self.steps_per_training_log == 0:
            self.writer.add_scalar('Train/Total', total_loss, self.step, avg_per_epoch = True)
            self.write_loss('Train/Color/', color_loss_track, avg_per_epoch = True)
            self.write_loss('Train/Polar/', polar_loss_track, avg_per_epoch = True)
            metrics = eval_all(full_color_polar, gt)
            for metric, value in metrics.items():
                metrics[metric] = value.mean().item()
                self.writer.add_scalar(f'Train/{metric}', metrics[metric], self.step, write_now = False, avg_per_epoch = True)

        if self.step % self.steps_per_epoch == 0:
            with torch.no_grad():
                print('Epoch average metrics:')
                self.writer.write_avg_epoch_metrics(self.step/self.steps_per_epoch)
                print('Validation set:')
                avg_val = eval_net(self.net, self.val_loader, self.writer, 'Val/', self.step/self.steps_per_epoch)
                self.update_learning_rate(avg_val)
                print('Test set:')
                eval_net(self.net, self.test_loader, self.writer, 'Test/', self.step/self.steps_per_epoch)
            self.save(avg_val)
            print(f"Done saving the checkpoint of epoch {int(self.step/self.steps_per_epoch)}!")
            print(self.ckp_dir)
            self.writer.flush()
        
    def update_learning_rate(self, val_metrics = None):
        if self.color_scheduler:
            if val_metrics and type(self.color_scheduler) == lr_scheduler.ReduceLROnPlateau:
                self.color_scheduler.step(val_metrics)
            else:
                self.color_scheduler.step()
        if self.polar_scheduler:
            if val_metrics and type(self.polar_scheduler) == lr_scheduler.ReduceLROnPlateau:
                self.polar_scheduler.step(val_metrics)
            else:
                self.polar_scheduler.step()

    def save(self, performance):
        opt_name = os.path.join(self.ckp_dir, 'optimizer.pt')
        current_epoch = self.step/self.steps_per_epoch
        net_name = os.path.join(self.ckp_dir, 'net_%08d.pt' % current_epoch)
        torch.save({'color_net': self.net.color_net.state_dict(),
                    'polar_net': self.net.polar_net.state_dict()}, net_name)
        torch.save({'color_opt': self.color_opt.state_dict(),
                    'polar_opt': self.polar_opt.state_dict()}, opt_name)
        final_checkpoint_file = os.path.join(self.ckp_dir, "final_checkpoint.txt")
        previous_checkpoint = os.path.join(self.ckp_dir, 'net_%08d.pt' % (current_epoch - 10))
        best_checkpoint = None
        best_performance = -float('inf')
        if os.path.isfile(final_checkpoint_file):
            with open(final_checkpoint_file) as f:
                best_checkpoint = f.read()
                best_epoch = int(best_checkpoint[-11:-3])
                best_performance = get_epoch_performance(self.log_dir, best_epoch)
            if current_epoch > 10 and best_checkpoint != previous_checkpoint and os.path.isfile(previous_checkpoint):
                os.remove(previous_checkpoint)
        if performance > best_performance:
            if best_checkpoint is not None:
                best_epoch = int(best_checkpoint[-11:-3])
                if current_epoch - best_epoch >= 10:
                    os.remove(best_checkpoint)
            with open(final_checkpoint_file, 'w') as f:
                f.write(net_name)


    def resume(self, hyperparameters, model_path=None, is_cpu=False):
        if not model_path:
            model_path = get_model_list(self.ckp_dir, "net")[-1] #get last model
        print(model_path)
        if is_cpu:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(model_path)
        epoch = int(model_path[-11:-3])
        self.step = epoch*self.steps_per_epoch
        self.net.color_net.load_state_dict(state_dict['color_net'])
        self.net.polar_net.load_state_dict(state_dict['polar_net'])
        optimizer_path = os.path.join(self.ckp_dir, 'optimizer.pt')
        if os.path.isfile(optimizer_path):
            if is_cpu:
                state_dict = torch.load(optimizer_path, map_location=torch.device('cpu'))
            else:
                state_dict = torch.load(optimizer_path)
            self.color_opt.load_state_dict(state_dict['color_opt'])
            self.polar_opt.load_state_dict(state_dict['polar_opt'])
        self.color_scheduler = get_scheduler(self.color_opt, hyperparameters['color_opt']['lr'], epoch)
        self.polar_scheduler = get_scheduler(self.polar_opt, hyperparameters['polar_opt']['lr'], epoch)
    
    def load_best(self, hyperparameters, is_cpu=False):
        final_checkpoint_file = os.path.join(self.ckp_dir, "final_checkpoint.txt")
        with open(final_checkpoint_file) as f:
            best_checkpoint = f.read()
            self.resume(hyperparameters, best_checkpoint, is_cpu)
        

class TwoStepSimultaneousTrainer(TwoStepTrainer):

    def update(self, raw_CPFA, sub_color_gt, gt):
        sub_color, full_color_polar = self.net(raw_CPFA)
        color_loss, color_loss_track, polar_loss, polar_loss_track = self.compute_loss(sub_color, sub_color_gt, full_color_polar, gt)
        total_loss = color_loss + polar_loss
        self.color_opt.zero_grad()
        self.polar_opt.zero_grad()
        total_loss.backward()
        self.color_opt.step()
        self.polar_opt.step()
        self.update_after(total_loss.item(), color_loss_track, polar_loss_track, full_color_polar, gt)  
