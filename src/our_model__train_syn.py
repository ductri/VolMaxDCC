import torch.nn as nn
import torch
import torchvision
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from our_model_training_utils_syn import sub_main, main


if __name__ == "__main__":
    # start = time.time()
    # list_m = range(1000, 10001, 1000)
    # # list_m = [200, 500, 800]
    # dataset_name = 'syn1-noise'
    # main(list_m, dataset_name)
    # end = time.time()
    # print(f'Total duration: {end-start:.2f}')


    lam = 0.0
    p = 0.1
    trial = 0
    list_hiddens = [3, 128, 3]
    dataset_name = 'syn1-noise'
    m = 5000
    sub_main(lam, False, p, trial, m, list_hiddens, dataset_name, epochs=1000, lr=1e-3)
    # m = 1000
    # sub_main(lam, True, p, trial, m, list_hiddens, dataset_name, epochs=300, lr=1e-3)

    # sub_main(lam, True, p, trial, m, list_hiddens, dataset_name, epochs=300, lr=1e-3)



