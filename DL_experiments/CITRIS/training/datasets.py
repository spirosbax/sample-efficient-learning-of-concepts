"""
PyTorch dataset classes for loading the datasets.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
import os
import json
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm


class Causal3DDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'pos-x': 'continuous_2',
        'pos-y': 'continuous_2',
        'pos-z': 'continuous_2',
        'rot-alpha': 'angle',
        'rot-beta': 'angle',
        'rot-gamma': 'angle',
        'rot-spot': 'angle',
        'hue-object': 'categ_8',
        'hue-spot': 'categ_8',
        'hue-back': 'categ_8',
        'obj-shape': 'categ_7',
        'obj-material': 'categ_3'
    })

    def __init__(self, data_folder, split='train', single_image=False, seq_len=2, coarse_vars=False, triplet=False, causal_vars=None, return_latents=False, img_width=-1, exclude_vars=None, max_dataset_size=-1, exclude_objects=None):
        super().__init__()
        filename = split
        if triplet:
            filename += '_triplets'
            if coarse_vars:
                filename += '_coarse'
        self.split_name = filename
        data_file = os.path.join(data_folder, f'{filename}.npz')
        if split.startswith('val') and not os.path.isfile(data_file):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
            data_file = os.path.join(data_folder, f'{filename.replace("val", "test")}.npz')
        assert os.path.isfile(data_file), f'Could not find causal3d dataset at {data_file}'
        arr = np.load(data_file)
        self.imgs = torch.from_numpy(arr['imgs'])[...,:3]
        if not triplet:
            self.imgs = self.imgs.permute(0, 3, 1, 2)
        else:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)
        if img_width > 0 and img_width != self.imgs.shape[-1]:
            full_shape = self.imgs.shape
            dtype = self.imgs.dtype
            if len(self.imgs.shape) == 5:
                self.imgs = self.imgs.flatten(0, 1)
            self.imgs = F.interpolate(self.imgs.float(), size=(img_width, img_width), mode='bilinear')
            self.imgs = self.imgs.reshape(full_shape[:-2] + (img_width, img_width))
            self.imgs = self.imgs.to(dtype)
        self.train = (split == 'train')
        self.single_image = single_image
        self.triplet = triplet
        self.coarse_vars = coarse_vars
        self.seq_len = seq_len if not (single_image or triplet) else 1
        self.return_latents = return_latents
        self.max_dataset_size = max_dataset_size
        self.encodings_active = False
        self._prepare_causal_vars(arr, coarse_vars, causal_vars, exclude_vars)
        self.sub_indices = torch.arange(self.imgs.shape[0] - self.seq_len + 1, dtype=torch.long)
        self.obj_triplet_indices = None
        if exclude_objects is not None:
            imgs_to_remove = torch.stack([self.true_latents[...,-1] == o for o in exclude_objects], dim=0).any(dim=0)
            if triplet:
                same_obj = (self.true_latents[:,:1,-1] == self.true_latents[:,:,-1]).all(dim=1)
                imgs_to_remove = torch.logical_and(~same_obj, imgs_to_remove.any(dim=1))
                self.obj_triplet_indices = (self.true_latents[:,-1:,-1] == torch.FloatTensor(exclude_objects)[None,:])
                self.obj_triplet_indices = (self.obj_triplet_indices.long() * torch.arange(1, 3, device=self.true_latents.device, dtype=torch.long)[None]).max(dim=-1).values
                
            for i in range(1, self.seq_len):
                imgs_to_remove[:-i] = torch.logical_or(imgs_to_remove[i:], imgs_to_remove[:-i])
            self.sub_indices = self.sub_indices[~imgs_to_remove[:self.sub_indices.shape[0]]]
            
            print(f'Removed images from {self.split_name} [objs {exclude_objects}]: {self.imgs.shape[0]} -> {self.sub_indices.shape[0]}')

    def _prepare_causal_vars(self, arr, coarse_vars=False, causal_vars=None, exclude_vars=None):
        target_names_l = list(Causal3DDataset.VAR_INFO.keys())
        targets = torch.from_numpy(arr['interventions'])
        true_latents = torch.cat([torch.from_numpy(arr['raw_latents']), torch.from_numpy(arr['shape_latents'])], dim=-1)
        assert targets.shape[-1] == len(target_names_l), f'We have {len(target_names_l)} target names, but the intervention vector has only {targets.shape[-1]} entries.'
        if not causal_vars:
            has_intv = torch.logical_and((targets == 0).any(dim=0), (targets == 1).any(dim=0))
            has_intv = torch.logical_and(has_intv, (true_latents[0:1] != true_latents).any(dim=0))
        else:
            has_intv = torch.Tensor([(n in causal_vars) for n in target_names_l]).bool()
        
        self.true_latents = true_latents[...,has_intv]
        self.target_names_l = [n for i, n in enumerate(target_names_l) if has_intv[i]]
        self.full_target_names = self.target_names_l[:]
        for i, name in enumerate(self.target_names_l):
            if name.startswith('hue') and Causal3DDataset.VAR_INFO[name].startswith('categ'):
                orig_shape = self.true_latents.shape
                self.true_latents = self.true_latents.flatten(0, -2)
                unique_vals, _ = torch.unique(self.true_latents[:,i]).sort()
                if unique_vals.shape[0] > 50:  # Replace categorical by angle
                    print(f'-> Changing {name} to angles')
                    Causal3DDataset.VAR_INFO[name] = 'angle'
                else:
                    num_categs = (unique_vals[None] == self.true_latents[:,i:i+1]).float().sum(dim=0)
                    unique_vals = unique_vals[num_categs > 10]
                    assert unique_vals.shape[0] <= int(Causal3DDataset.VAR_INFO[name].split('_')[-1])
                    self.true_latents[:,i] = (unique_vals[None] == self.true_latents[:,i:i+1]).float().argmax(dim=-1).float()
                self.true_latents = self.true_latents.reshape(orig_shape)

        if exclude_vars is not None:
            for v in exclude_vars:
                assert v in target_names_l, f'Could not find \"{v}\" in the name list: {target_names_l}.'
                idx = target_names_l.index(v)
                if self.triplet and (targets[:,idx] == 1).any():
                    print(f'[!] WARNING: Triplets will not work properly for \"{v}\"')
                has_intv[idx] = False

        self.targets = targets[...,has_intv]
        self.target_names_l = [n for i, n in enumerate(target_names_l) if has_intv[i]]
        if coarse_vars:
            for abs_class in ['rot', 'pos']:
                abs_vars = torch.Tensor([n.startswith(f'{abs_class}-') and not n.endswith('spot') for n in self.target_names_l]).bool()
                self.targets = torch.cat([self.targets[...,abs_vars].any(dim=-1, keepdims=True), self.targets[...,~abs_vars]], dim=-1)
                self.target_names_l = [abs_class] + [n for i, n in enumerate(self.target_names_l) if not abs_vars[i]]
        print(f'Considering the following causal variables: {self.target_names_l}')

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=256):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-1]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        causal_var_info = OrderedDict()
        for key in self.target_names_l:
            causal_var_info[key] = Causal3DDataset.VAR_INFO[key]
        return causal_var_info

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def __len__(self):
        if self.max_dataset_size > 0:
            return min(self.max_dataset_size, self.sub_indices.shape[0])
        else:
            return self.sub_indices.shape[0]

    def __getitem__(self, idx):
        idx = self.sub_indices[idx]
        if self.single_image:
            img = self.imgs[idx]
            img = self._prepare_imgs(img)
            if self.return_latents:
                lat = self.true_latents[idx]
                return img, lat
            else:
                return img
        elif self.triplet:
            imgs = self.imgs[idx]
            targets = self.targets[idx]
            imgs = self._prepare_imgs(imgs)
            if self.return_latents:
                lat = self.true_latents[idx]
                if self.obj_triplet_indices is not None:
                    return imgs, targets, lat, self.obj_triplet_indices[idx]
                else:
                    return imgs, targets, lat
            else:
                return imgs, targets
        else:
            imgs = self.imgs[idx:idx+self.seq_len]
            targets = self.targets[idx:idx+self.seq_len-1]
            imgs = self._prepare_imgs(imgs)
            if self.return_latents:
                lat = self.true_latents[idx:idx+self.seq_len]
                return imgs, targets, lat
            else:
                return imgs, targets

