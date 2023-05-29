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
    m = 8994
    lam = 1e-2
    p = 'real'
    trial = 0
    is_B_trainable = False

    dataset_name = 'imagenet10-cc-10k'
    list_hiddens = [512, 512, 10]

    sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs=100, \
            lr=1e-3, num_workers=0, debug=True, optimizer_name='adam')

