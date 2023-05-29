import torch.nn as nn
import torch
import torchvision
import argparse
import socket
import time
from joblib import Parallel, delayed

from our_model_training_utils import sub_main, main


if __name__ == "__main__":
    start = time.time()

    m = 10000
    dataset_name = 'cifar10-byol-15k'
    list_list_hiddens = [[512, 512, 10],]
    list_lam = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    list_p = ['-1', 'b', 'd', 'e']
    num_trials = 5
    lr = 1e-3
    epochs = 200
    main(m, dataset_name, list_p, num_trials, lr, epochs, list_list_hiddens)
    end = time.time()
    print(f'Total duration: {end-start:.2f}')

