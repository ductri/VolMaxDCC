import torch.nn as nn
import torch
import os
import numpy as np
import torch
import torchvision
import argparse
from joblib import Parallel, delayed
import pickle as pkl

from utils import aux_tools


def sub_main(m, p, trial):
    data_loader, dataset = aux_tools.load_data('imagenet10-cc-10k', batch_size=128, \
            num_workers=0, m=m, include_dataset=True, p=p, trial=trial)
    class_num = 10


def main():
    m = 8994
    for i in range(1):
        sub_main(m, 'real', i)
    for i in range(1):
        sub_main(m, '-1', i)
    # for i in range(5):
    #     sub_main(-1, i)
    # for i in range(5):
    #     sub_main('c', i)
    # for i in range(5):
    #     sub_main(-1, i)

def inspect():
    num_trials = 1
    m = 8994
    dataset_name = 'imagenet10-cc-10k'

    for p in ['-1', 'real']:
        rates = []
        for trial in range(num_trials):
            pair_path = f'datasets/imagenet10/pairs/pair_{m:d}_{str(p):s}_{dataset_name}_trial_{trial}.pkl'
            with open(pair_path, 'rb') as file_handler:
                data_dict = pkl.load(file_handler)
            flipping_rate = 1 - (data_dict['true_label_pairs'] == \
                    data_dict['label_pairs']).mean()
            rates.append(flipping_rate)
        print(f'p={p} rate={np.mean(rates)}')


if __name__ == "__main__":
    main()
    inspect()

