from torch.optim import lr_scheduler
from dataset import BaseTestingDataset, PatchTrainDataset
from torch.utils.data import DataLoader
import os
import yaml
import shutil
import torch.nn.init as init
from CPFA import CPFA
import torch
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def get_model_name_from_config_path(cfg_path):
    model_name = os.path.splitext(os.path.basename(cfg_path))[0]
    return model_name

def scale(x, old_min, old_max, new_min, new_max):
    return (x - old_min)*(new_max - new_min)/(old_max - old_min) + new_min

def compute_stokes(x):
    # dim info of x: (batch, polar, rgb,  h, w)
    # dim info here: polar = 0 => 0, polar = 1 => 45, polar = 2 => 90, polar = 3 => 135
    s = torch.stack([torch.sum(x, axis=1)/2,
                   x[:,0,:,:,:]-x[:,2,:,:,:],
                   x[:,1,:,:,:]-x[:,3,:,:,:]],
                    axis=1)

    # now let us scale the Stokes parameters
    scaled_s = torch.stack([scale(s[:,0,:,:,:], 0, 2, 0, 1),
                          scale(s[:,1,:,:,:], -1, 1, 0, 1),
                          scale(s[:,2,:,:,:], -1, 1, 0, 1)],
                           axis=1)
    return s, scaled_s

def compute_dop(s, d = 0):
    # dim info of s: (batch, S, rgb, h, w)
    dop = torch.div(torch.sqrt(torch.square(s[:,1,:,:,:]) + torch.square(s[:,2,:,:,:]) + d), s[:,0,:,:,:])

    # now let us scale the DOP
    #dop = scale(dop, 0, 2, 0, 1)
    # we do not need to scale the DOP here because we divide by 2 instead of 4
    # reference: https://github.com/pjlapray/Polarimetric_Spectral_Database/issues/2
    dop[dop < 0] = 0
    dop[dop > 1] = 1
    dop[torch.isnan(dop)] = 0
    dop[torch.isinf(dop)] = 0

    return dop

def compute_aop(s):
    # dim info of s: (batch, S, rgb, h, w)

    aop = 0.5*torch.atan2(s[:,2,:,:,:], s[:,1,:,:,:])

    # torch.atan2 function ranges from -pi to pi
    # try: torch.rad2deg(torch.atan2(torch.tensor(-0.000000001), torch.tensor(-0.999999)))
    # and: torch.rad2deg(torch.atan2(torch.tensor(0.000000001), torch.tensor(-0.999999)))
    # ref: https://www.mathworks.com/help/matlab/ref/atan2.html

    # now let us scale the AOP
    aop = scale(aop, np.deg2rad(-90), np.deg2rad(90), 0, 1)

    return aop

# reference: https://discuss.pytorch.org/t/how-to-change-a-batch-rgb-images-to-ycbcr-images-during-training/3799
def rgb_to_ycbr(x):
    # dim info of x: (batch, polar, rgb, h, w)
    y = torch.zeros_like(x)
    y[:,:,0,:,:] = x[:,:,0,:,:]*65.481 + x[:,:,1,:,:]*128.553 + x[:,:,2,:,:]*24.966 + 16
    y[:,:,1,:,:] = -x[:,:,0,:,:]*37.797 - x[:,:,1,:,:]*74.203 + x[:,:,2,:,:]*112 + 128
    y[:,:,2,:,:] = x[:,:,0,:,:]*112 - x[:,:,1,:,:]*93.786 - x[:,:,2,:,:]*18.214 + 128
    return y
    

#In case we run our testing scheme using PatchDataset, we need to stitch the patch estimations
def stitch(x, h, w, padding=0): 
    # dim info of x: (batch, polar, rgb, h_patch, w_patch)
    output = torch.zeros(1, x.shape[1], x.shape[2], h, w).to(device=device)
    centre_patch_size = x.shape[4] - padding*2
    patch_w_n = int(w / centre_patch_size)
    for idx, x_b in enumerate(x):
        i = int(idx / patch_w_n)
        j = int(idx % patch_w_n)
        left = int(centre_patch_size*j)
        top = int(centre_patch_size*i)
        output[0, :, :, top:top+centre_patch_size, left:left+centre_patch_size] = x_b[:, :, :, padding:x.shape[3]-padding, padding:x.shape[4]-padding]
    return output

def get_optimizer(config, params):
    if config['optimizer'].lower() == 'adam':
        lr = config['lr']['init']
        beta1 = config['beta1']
        beta2 = config['beta2']
        weight_decay = config['weight_decay']
        opt = torch.optim.Adam([p for p in params], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    '''if config['optimizer'].lower() == 'radam':
        lr = config['lr']['init']
        beta1 = config['beta1']
        beta2 = config['beta2']
        weight_decay = config['weight_decay']
        opt = optim.RAdam([p for p in params], lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)'''
    return opt

def get_CPFA(config):
    pattern = [x.strip() for x in config['pattern'].split(',')]
    if len(pattern) == 1:
        cpfa = CPFA(pattern[0], config['max_height'], config['max_width'])
    else:
        cpfa = CPFA(pattern[0], config['max_height'], config['max_width'], pattern[1])
    return cpfa

'''
hsv_to_rgb function is written by PolarNick239: https://gist.github.com/PolarNick239/691387158ff1c41ad73c
'''

def hsv_to_rgb(hsv):
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

    return rgb.reshape(input_shape)

'''
Below functions are  copies with some possible modifications from: https://github.com/NVlabs/MUNIT/blob/master/utils.py
'''

def get_scheduler(optimizer, hyperparameters, last_epoch=-1):
    if 'policy' not in hyperparameters or hyperparameters['policy'] == 'constant':
        scheduler = None
    elif hyperparameters['policy'] == 'stepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=last_epoch)
    elif hyperparameters['policy'] == 'reduceLRonplateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)        
        

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def get_all_data_loaders(config):
    patches_per_image = config['patches_per_image']
    num_workers = config['num_workers']
    root = config['data_root']
    patch_size = config['patch_size'] if 'patch_size' in config else 0
    batch_size = config['number_of_images']
    hdf5_root = os.path.join(root, 'hdf5')
    train_flist = os.path.join(root, 'train.txt')
    val_flist = os.path.join(root, 'val.txt')
    test_flist = os.path.join(root, 'test.txt')
    cpfa = get_CPFA(config)
    aug_cfg = config['augmentation'] if 'augmentation' in config else None
    
    if patch_size > 0:
        train_dataset = PatchTrainDataset(hdf5_root, train_flist, cpfa, patch_size, patches_per_image, aug_cfg)
        trainv_dataset = BaseTestingDataset(hdf5_root, train_flist, cpfa)
    else:
        train_dataset = BaseTrainDataset(hdf5_root, train_flist, cpfa)
        trainv_dataset = BaseTestingDataset(hdf5_root, train_flist, cpfa)
    val_dataset = BaseTestingDataset(hdf5_root, val_flist, cpfa)
    test_dataset = BaseTestingDataset(hdf5_root, test_flist, cpfa)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True, num_workers=num_workers)
    trainv_loader = DataLoader(dataset=trainv_dataset, batch_size=1, shuffle=False, drop_last = False, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, drop_last = False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last = False, num_workers=num_workers)
    return train_loader, trainv_loader, val_loader, test_loader

def prepare_sub_folder(output_directory, resume, model_name):
    log_dir = os.path.join('./logs', model_name)
    csv_log_file = f'./logs/{model_name}.csv'
    if not resume and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    if not resume and os.path.isfile(csv_log_file):
        os.remove(csv_log_file)
    output_directory = os.path.join(output_directory, model_name)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    elif not resume:
        shutil.rmtree(checkpoint_directory)
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, log_dir

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    return gen_models

def get_checkpoint(dirname, key):
    fpath = os.path.join(dirname, 'final_checkpoint.txt')
    if os.path.exists(fpath):
        with open(fpath) as f:
            return f.read()
    return get_model_list(dirname, key)[-1]

def get_epoch_performance(log_file, epoch):
    tag = 'Val/CPSNR/Average'
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()
    df = pd.DataFrame(event_acc.Scalars(tag),columns=['wall_time', 'epoch', 'value'])
    df = df[df.epoch == epoch]
    df = df.drop('wall_time', axis=1)
    return df.value.to_numpy()[0]

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'zero':
                init.constant_(m.weight.data, 0.0)
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
                
    return init_fun
