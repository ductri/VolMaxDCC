import torch.nn as nn
import torch
import os
import numpy as np
import torch
import torchvision
import argparse
from joblib import Parallel, delayed

from utils import aux_tools


def sub_main(m, p, trial):
    data_loader, dataset = aux_tools.load_data('stl10-cc', batch_size=128, \
            num_workers=0, m=m, include_dataset=True, p=p, trial=trial)
    class_num = 10



def main():
    list_M = [10000,]
    for m in list_M:
        for i in range(5):
            sub_main(m, -1, i)
    # for i in range(5):
    #     sub_main(3, i)
    # for i in range(5):
    #     sub_main(4, i)
    # for i in range(5):
    #     sub_main(5, i)


if __name__ == "__main__":
    main()

