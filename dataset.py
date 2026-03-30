import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import Resize

from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

from abc import abstractmethod, ABC
from einops import rearrange
import warnings
from copy import deepcopy

import utils


class DfncDataset(Dataset):
    def __init__(self, file_path, label_path, target, args, num_frames=-1, format='2D', norm=False, augment=False): # augment added on 23/1/26
        super().__init__()
        self.format = format
        self.data = torch.tensor(np.load(file_path), dtype=torch.float32)
        self.labels = torch.tensor(np.load(label_path), dtype=torch.float32)

        # Resize FC matrices to 128x128
        if self.data.ndim == 4:  # FC data (N, H, W, C)
            resize = Resize((128, 128))
            self.data = torch.stack([resize(d) for d in self.data])
            # Add time dimension: (N, C, H, W, 1)
            self.data = self.data.unsqueeze(-1)

        # Normalize FC matrices to zero mean and unit variance
        self.data = (self.data - self.data.mean()) / (self.data.std() + 1e-8)

        assert self.data.shape[0] == self.labels.shape[0]

        if 'HCP' in file_path:
            target_map = {'sex': 0, 'age': 1, 'fluid_unAdj': 2, 'fluid_ageAdj': 3}
        elif 'EMOTION' in file_path:
            target_map = {'sex': 0, 'age': 1, 'fluid_cognition': 2}
        elif 'WM' in file_path:
            target_map = {'sex': 0, 'age': 1, 'fluid_cognition': 2}
        elif 'EHBS' in file_path:
            target_map = {'age': 0, 'sex': 1, 'neuroscopy': 2, 'AsymAD': 3}
        elif 'ADNI' in file_path:
            target_map = {'age': 0, 'sex': 1, 'AD': 2}
        elif 'UKB' in file_path:
            target_map = {'sex': 0, 'age': 1, 'fluid': 2}
        else:
            raise NotImplementedError

        if args.num_classes > 1:  # non-binary
            self.labels = self.labels[:, target_map[target]].to(torch.int64)
        else:
            self.labels = self.labels[:, target_map[target]][:, None]

        # Normalize regression targets (fluid_cognition)
        if args.task == 'regression' and target == 'fluid_cognition':
            self.labels = (self.labels - self.labels.mean()) / (self.labels.std() + 1e-8)

        if num_frames != -1:
            self.data = self.data[:, :num_frames, ...]
        self.data = self.data.mean(1, keepdim=True) if 'mean' in args.dataset_name else self.data

    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, index):
        data, label = self.data[index].unsqueeze(0), self.labels[index]
        if self.format == '1DT':  # B T L
            data = data.permute(0, 2, 1).unsqueeze(1)  # B C L T
        elif self.format == '2DT':  # target: B C H W T
            # Legacy dfnc: (B, T, H, W) — permute to (B, H, W, T), then add channel.
            # WM / FC from prepare_wm_data: (N, C, H, W, T) with C=1 — after batch dim (B, C, H, W, T), no permute.
            # Older path: (N, H, W, C, T) — (B, H, W, C, T) needs reorder to B,C,H,W,T.
            if data.dim() == 4:
                data = data.permute(0, 2, 3, 1).unsqueeze(1)  # B C H W T
            elif data.dim() == 5:
                _, d1, d2, d3, _ = data.shape
                if d1 <= 8 and d2 >= 8 and d3 >= 8:
                    # (B, C, H, W, T) with channel first (e.g. C=1, H=W=128)
                    pass
                elif d1 >= 8 and d2 >= 8:
                    # (B, H, W, C, T)
                    data = data.permute(0, 3, 1, 2, 4).contiguous()
                else:
                    raise ValueError(
                        f"2DT: cannot infer layout for shape {tuple(data.shape)}; "
                        "expected (B,T,H,W), (B,C,H,W,T), or (B,H,W,C,T)."
                    )
            else:
                raise ValueError(f"2DT: expected 4D or 5D batch, got dim={data.dim()} shape={tuple(data.shape)}")
        elif self.format == 'T2D':
            data = data.unsqueeze(1)  # B C T H W
        elif self.format == 'graph':
            data = torch.tensor(utils.vec2mat(data))
        return data.squeeze(0), label
    
    @property
    def cls_num(self):
        return [self.__len__() - torch.count_nonzero(self.labels), torch.count_nonzero(self.labels)]


def get_data(fold_id, args, data_name, fold=5, is_ddp=False, is_test_ddp=False, target=None, format='2D'):
    if 'EMOTION_RL' in data_name:  # EMOTION_RL must come before EMOTION
        file_path = 'dataset/EMOTION_RL/emotion_rl_fc_combined.npy'
        label_path = 'dataset/EMOTION_RL/emotion_rl_labels.npy'
    elif 'EMOTION' in data_name:  # EMOTION_LR (default)
        file_path = 'dataset/EMOTION_LR/emotion_lr_fc_combined.npy'
        label_path = 'dataset/EMOTION_LR/emotion_lr_labels.npy'
    elif 'WM_RL' in data_name:  # WM_RL must come before WM
        file_path = 'dataset/WM_RL/wm_rl_fc_combined.npy'
        label_path = 'dataset/WM_RL/wm_rl_labels.npy'
    elif 'WM' in data_name:  # WM_LR (default)
        file_path = 'dataset/WM_LR/wm_lr_fc_resized.npy'
        label_path = 'dataset/WM_LR/wm_lr_labels.npy'
    elif 'fnc' in data_name:
        if 'HCP' in data_name:
            if '1D' in format or 'graph' in format:
                file_path = 'data/HCP/dfnc/dfnc_hcp_REST1_RL.npy'
            else:
                file_path = '../data/HCP/dfnc/dfnc_hcp_REST1_RL_894*64*64.npy'
            label_path = '../data/HCP/dfnc/labels_hcp_REST1_RL.npy'
        elif 'ADNI' in data_name:
            if '1D' in format or 'graph' in format:
                file_path = 'data/ADNI/dfnc/dfnc_ADNI_maxTR_10.npy'
            else:
                file_path = 'data/ADNI/dfnc/dfnc_ADNI_maxTR_10_92*64*64.npy'
            label_path = 'data/ADNI/dfnc/labels_dfnc_ADNI.npy'
        elif 'UKB' in data_name:
            if '1D' in format or 'graph' in format:
                file_path = 'data/UKB/dfnc/dfnc_ukb_fl.npy'
                label_path = 'data/UKB/dfnc/labels_ukb_fl.npy'
            elif 'sfnc' in data_name:
                file_path = 'data/UKB/sfnc/sfnc_ukb_fl_1*64*64.npy'
                label_path = 'data/UKB/sfnc/label_sfnc_fl.npy'
            elif 'dfnc' in data_name:
                file_path = 'data/UKB/dfnc/dfnc_ukb_fl_450*64*64.hdf5'
                label_path = 'data/UKB/dfnc/labels_ukb_fl.npy'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    norm = True if 'norm' in data_name else False

    dataset = DfncDataset(file_path, label_path, target=target, args=args, format=format, norm=norm)

    if fold == -1:  # use all data
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=16, pin_memory=True)
        return dataloader

    skf = StratifiedKFold(n_splits=fold) if args.task == 'classification' else KFold(n_splits=fold)
    indices = np.arange(len(dataset))
    for i, (train_indices, test_indices) in enumerate(skf.split(indices, dataset.labels)):
        if i == fold_id:
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            test_subset = torch.utils.data.Subset(dataset, test_indices)

            if is_ddp or is_test_ddp:
                raise NotImplementedError

            if 'dgl_graph' in format:
                assert is_ddp == False, 'currently no implementation for DDP of graph models'
                dataloader = GraphDataLoader
            else:
                dataloader = torch.utils.data.DataLoader

            is_shuffle = True if 'hdf5' not in file_path else False

            data_loader_train = dataloader(train_subset, shuffle=is_shuffle, batch_size=args.batch_size, num_workers=16, pin_memory=True)
            data_loader_test = dataloader(test_subset, shuffle=False, batch_size=args.batch_size, num_workers=16, pin_memory=True)
            return data_loader_train, data_loader_test