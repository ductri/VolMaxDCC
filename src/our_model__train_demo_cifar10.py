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
    lam = 1e-3
    p = 'e'
    trial = 1
    is_B_trainable = True

    dataset_name = 'cifar10-byol-15k'
    list_hiddens = [512, 2048, 10]

    sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs=100, \
            lr=1e-3, num_workers=2, debug=True, optimizer_name='adam')

    # Parallel(n_jobs=5)(delayed(sub_main)(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs=100, \
    #         lr=1e-3, num_workers=0, debug=True) for lam in [1e-2, 0.0])

