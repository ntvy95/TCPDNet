import torch.utils.data as data
import h5py
import torch.nn.functional as F
import torch
import os
import transforms

def default_flist_reader(path):
    with open(path) as f:
        return f.read().splitlines()

def default_file_reader(path):
    f = h5py.File(path, 'r')
    gt = torch.from_numpy(f['gt'][()]).to(dtype=torch.float)
    return gt

class BaseDataset(data.Dataset):
    def __init__(self, root, flist, CPFA,
                 flist_reader = default_flist_reader,
                 file_reader = default_file_reader):

        self.CPFA = CPFA
        self.root = root
        self.flist = flist_reader(flist)
        self.file_reader = file_reader

    def __len__(self):
        return len(self.flist)

class BaseTestingDataset(BaseDataset):
    def __getitem__(self, idx):
        gt = self.file_reader(os.path.join(self.root, self.flist[idx]))
        raw_CPFA = self.CPFA.generate_raw_CPFA(gt)
        sub_color_gt = self.CPFA.sub_color_image(gt)
        return idx, raw_CPFA, gt, sub_color_gt

class BaseTrainDataset(BaseDataset):
    def __getitem__(self, idx):
        gt = self.file_reader(os.path.join(self.root, self.flist[idx]))
        raw_CPFA = self.CPFA.generate_raw_CPFA(gt)
        sub_color_gt = self.CPFA.sub_color_image(gt)
        return raw_CPFA, gt, sub_color_gt

class PatchDataset(BaseDataset):
    def __init__(self, root, flist, CPFA,
                 patch_size = 64,
                 flist_reader = default_flist_reader,
                 file_reader = default_file_reader):
        super().__init__(root, flist, CPFA, flist_reader, file_reader)
        self.patch_size = patch_size    

class PatchTrainDataset(PatchDataset):
    def __init__(self, root, flist, CPFA,
                 patch_size = 64,
                 patches_per_image = 10,
                 augmentation_cfg = None,
                 flist_reader = default_flist_reader,
                 file_reader = default_file_reader):
        super().__init__(root, flist, CPFA, patch_size, flist_reader, file_reader)
        self.patches_per_image = patches_per_image
        self.transform = [transforms.RandomCrop(patch_size)]
        self.random_CPFA = False
        if augmentation_cfg:
            aug_ops = augmentation_cfg['ops'].split(',')
            print(f"Data augmentation")
            for op in aug_ops:
                op = op.strip()
                if op == 'rot90':
                    self.transform.append(transforms.Random90Rotation())
                    print("- Random rotation")
        self.transform = transforms.Compose(self.transform)

            
    def __getitem__(self, idx):
        gt = self.file_reader(os.path.join(self.root, self.flist[idx]))
        crops_raw_CPFA = []
        crops_gt = []
        crops_sub_color_gt = []
        for i in range(self.patches_per_image):
            c_gt = self.transform(gt)
            pattern_starting_point = torch.tensor([0,0])
            c_raw_CPFA = self.CPFA.generate_raw_CPFA(c_gt, pattern_starting_point)
            c_sub_color_gt = self.CPFA.sub_color_image(c_gt)
            crops_raw_CPFA.append(c_raw_CPFA)
            crops_gt.append(c_gt)
            crops_sub_color_gt.append(c_sub_color_gt)
            if self.random_CPFA:
                crops_pattern_starting_point.append(pattern_starting_point)
        raw_CPFA = torch.stack(crops_raw_CPFA, axis=0)
        gt = torch.stack(crops_gt, axis=0)
        sub_color_gt = torch.stack(crops_sub_color_gt, axis=0)
        return raw_CPFA, gt, sub_color_gt
            
        

        
        
