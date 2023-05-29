import torch.nn as nn
import torch
import os
import numpy as np
import torch
import torchvision
import argparse
from datetime import datetime
import time
from tqdm import tqdm
from joblib import Parallel, delayed

from utils import  aux_tools
from my_dataset import StandardDataset


class ImageNet10Oracle(nn.Module):
    def __init__(self, input_size, class_num):
        super(ImageNet10Oracle, self).__init__()
        self.normalize_projector = nn.Sequential(
            nn.Linear(input_size, class_num),
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.class_num = class_num

    def forward(self, x):
        return self.normalize_projector(x)

    def compute_loss(self, logits_pred, y):
        return self.loss_fn(logits_pred, y)


def train(model, optimizer, data_loader, model_path, args):
    counter = 0
    for epoch in range(args.epochs):
        loss_epoch = 0
        loss1_epoch = 0
        loss2_epoch = 0
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

            if step % 1 == 0:
                aux_tools.save_model2(model_path, model, optimizer, step)
            print(f'Epoch [{epoch}/{args.epochs}] \t Loss: {loss_epoch / len(data_loader):.5} Acc: {acc_epoch / len(data_loader):.5}')
    aux_tools.save_model2(model_path, model, optimizer, args.epochs)

def sub_main():
    START = time.time()
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.epochs = 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    X = np.load('./datasets/imagenet-10/feature_600.npy')[:100,:]
    y = np.load('./datasets/imagenet-10/label_600.npy')[:100]
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    dataset = StandardDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=25)
    class_num = 10

    model = ImageNet10Oracle(512, class_num)
    model = model.to('cuda')
    print('Number of trainable var: %d\n' % aux_tools.count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    code_name = 'imagenet10-oracle-e600'
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

