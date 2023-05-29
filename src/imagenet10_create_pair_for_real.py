import torch.nn as nn
import torch
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import argparse
import pickle as pkl
from PIL import Image
import pandas as pd
from tqdm import tqdm

from utils import aux_tools


# class ShuffledDataset(Dataset):
#     def __init__(self, dataset, shuffle_inds):
#         self.dataset = dataset
#         self.shuffle_inds = shuffle_inds
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, ind):
#         return self.dataset[self.shuffle_inds[ind]]
#
#     def get_url(self, ind):
#         __import__('pdb').set_trace()
#         return self.imgs[self.shuffle_inds[ind]]


def main():
    ##########################################
    ### DANGER !!!! ####
    ### REALLY CAREFUL WHEN RUNNING THIS CODE
    ##########################################
    # dataset = torchvision.datasets.ImageFolder(
    #     root='datasets/imagenet-10/raw'
    # )
    # meta_data = {}
    # shuffle_inds = np.arange(len(dataset))
    # np.random.shuffle(shuffle_inds)
    # meta_data['shuffle_inds'] = shuffle_inds
    # N = 10000
    # m = 15000
    # i1 = np.random.randint(0, N, size=m)
    # i2 = np.random.randint(0, N, size=m)
    # meta_data['i1'] = i1
    # meta_data['i2'] = i2
    # meta_data['i1_path'] = [dataset.imgs[i] for i in i1]
    # meta_data['i2_path'] = [dataset.imgs[i] for i in i2]
    # meta_data['N'] = N
    # meta_data['m'] = m
    # with open('./datasets/imagenet-10/real/meta_data.pkl', 'wb') as o_f:
    #     pkl.dump(meta_data, o_f)
    # We should have enough info to construst pairs of images now.
    #
    ### End of DANGER !!!! ####
    ##########################################

    # dataset = torchvision.datasets.ImageFolder(
    #     root='datasets/imagenet-10/raw',
    #     transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), \
    #         ])
    # )
    # with open('datasets/imagenet-10/real/meta_data.pkl', 'rb') as i_f:
    #     meta_data = pkl.load(i_f)
    # shuffle_inds = meta_data['shuffle_inds']
    #
    # list_filenames = []
    # true_pairwise_labels = []
    # for i1, i2, i1_path, i2_path in tqdm(zip(meta_data['i1'], meta_data['i2'], meta_data['i1_path'], meta_data['i2_path']), total=len(meta_data['i1'])):
    #     i1_path = i1_path[0].replace('/', '_')[:-5]
    #     i2_path = i2_path[0].replace('/', '_')[:-5]
    #     dst = Image.new('RGB', (224*2 + 10, 224), (0,0,0))
    #     dst.paste(dataset[shuffle_inds[i1]][0], (0, 0))
    #     dst.paste(dataset[shuffle_inds[i2]][0], (224+10, 0))
    #     filename = f'{i1}-{i2}-{i1_path}-{i2_path}.jpeg'
    #     dst.save(f'datasets/imagenet-10/real/upload_pairs/{filename}')
    #
    #     list_filenames.append(filename)
    #     true_pairwise_labels.append((dataset[shuffle_inds[i1]][1] == dataset[shuffle_inds[i2]][1])*1.0)
    #
    # df = pd.DataFrame({'image_url': list_filenames, 'true_pairwise_labels': true_pairwise_labels})
    # df.to_csv('datasets/imagenet-10/real/meta_pairwise.csv', index=None)



if __name__ == "__main__":
    main()

