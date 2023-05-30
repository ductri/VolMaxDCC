import random
import os
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle as pkl
import pandas as pd

import stl10_oracle__train
import cifar10_oracle__train
from imagenet10_oracle__train import ImageNet10Oracle
import oracle_cifar10

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(name, batch_size=32, num_workers=8):
    if name=='imagenet10-cc-10k':
        pair_path = f'datasets/imagenet10/pairs/pair_10000_trial_0.pkl'
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = pkl.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        label_pairs = label_pairs.astype(np.float32)
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        true_y = data_dict['true_y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        raise Exception('typo')

    data_loader = torch.utils.data.DataLoader(
        data_pairs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True)
    return data_loader

class PairDataset(Dataset):
    def __init__(self, ind_pairs, label_pairs, X, others=[]):
        self.ind_pairs = ind_pairs
        self.label_pairs = label_pairs
        self.X = torch.from_numpy(X)
        self.others = others

    def __len__(self):
        return len(self.ind_pairs)

    def __getitem__(self, idx):
        i1, i2 = self.ind_pairs[idx]
        return (self.X[i1], self.X[i2]), self.label_pairs[idx]

    def getitem_extra(self, idx):
        i1, i2 = self.ind_pairs[idx]
        return (self.X[i1], self.X[i2]), self.label_pairs[idx], (i1, i2, self.others[idx])

def save_model2(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
    print('Save model to %s' % model_path)

