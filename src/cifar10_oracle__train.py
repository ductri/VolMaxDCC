import os
import numpy as np
import argparse
from datetime import datetime
import time
import pickle as pkl

import torchvision
import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from joblib import Parallel, delayed

from my_dataset import StandardDataset
from utils import aux_tools


class CIFAR10Oracle(nn.Module):
    def __init__(self, input_size, class_num):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.transforms = weights.transforms()

        self.backbone = torchvision.models.resnet18()
        self.normalize_projector = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, class_num),
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.class_num = class_num

    def forward(self, x):
        return self.normalize_projector(x)

    def compute_loss(self, logits_pred, y):
        return self.loss_fn(logits_pred, y)


def main_loop(model, optimizer, data_loader, model_path, epochs):
    counter = 0
    for epoch in range(epochs):
        loss_epoch = 0
        acc_epoch = 0
        trange = tqdm(enumerate(data_loader), total=len(data_loader))
        for step, (x, y) in trange:
            counter += 1
            optimizer.zero_grad()
            x = x.to('cuda')
            y = y.to('cuda')
            logits_pred = model(x)
            loss = model.compute_loss(logits_pred, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            with torch.no_grad():
                acc = ((logits_pred.argmax(1) == y)*1.0).mean()
                acc_epoch += acc

        if epoch % 5 == 0:
            aux_tools.save_model2(model_path, model, optimizer, epoch)


def sub_main():
    START = time.time()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 1

    args.seed = 11
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    # with open('datasets/cifar10/byol/features.p', 'rb') as i_f:
    #     data = pkl.load(i_f)
    #     X = data[0]
    #     y = data[1]

    with open('datasets/cifar10/simclr_pytorch/cifar10_train.pkl', 'rb') as i_f:
        data = pkl.load(i_f)
        X = data['X'][:1000]
        y = data['Y'][:1000]

    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    dataset = StandardDataset(X, y)
    print(f'Training size: {len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128)
    class_num = 10

    model = CIFAR10Oracle(2048, class_num)
    model = model.to('cuda')
    print('Number of trainable var: %d\n' % aux_tools.count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    code_name = f'oracle/cifar10-simclr_pytorch'
    model_path = 'save/%s'%code_name
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # train
    train(model, optimizer, data_loader, model_path, args)

    END = time.time()
    print('elapsed time: %f s' %  (END - START))

def main():
    sub_main()

if __name__ == "__main__":
    main()

