from collections import OrderedDict
import torch.nn as nn
import torch
import os
import pickle
import numpy as np
import argparse
from datetime import datetime
import time
from tqdm import tqdm
from joblib import Parallel, delayed

from utils import aux_tools
from our_model import OurModel, OurModelHacking


def train(model, optimizer, data_loader, writer, model_path, lam, X, args):
    for epoch in range(args.epochs):
        loss_epoch = 0
        loss1_epoch = 0
        loss2_epoch = 0
        acc_epoch = 0
        trange = tqdm(enumerate(data_loader), total=len(data_loader))

        if epoch % 5 == 0:
            aux_tools.save_model2(model_path, model, optimizer, epoch)
        for step, ((x_i, x_j), y) in trange:
            optimizer.zero_grad()
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)
            y = y.to(args.device)
            c_i, c_j = model.forward(x_i, x_j)
            loss, loss1, loss2 = model.compute_loss(model._get_B(), c_i, c_j, y, lam, X)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            loss1_epoch += loss1.item()
            loss2_epoch += loss2.item()
            with torch.no_grad():
                prob_pred = model.link_prediction(model._get_B(), c_i, c_j)
                acc = aux_tools.get_accuracy(prob_pred>=0.5, y)
                acc_epoch += acc


        print(f"Epoch [{epoch}/{args.epochs}] \t Loss: {loss_epoch / len(data_loader):.5} Loss1: {loss1_epoch / len(data_loader):.5} Loss2: {loss2_epoch / len(data_loader):.5} Acc: {acc_epoch / len(data_loader):.5} lam={lam:.5}")
    aux_tools.save_model2(model_path, model, optimizer, args.epochs)

def sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs=100, lr=1e-2):
    START = time.time()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs=epochs
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args.seed = 101
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    data_loader, dataset = aux_tools.load_data(dataset_name, batch_size=256, \
            num_workers=0, m=m, include_dataset=True, p=p, trial=trial)
    class_num = 3


    list_layers = [('linear1', nn.Linear(3, 10)), ('relu1', nn.ReLU()), \
            ('linear2', nn.Linear(10, 128))]
    backbone = nn.Sequential(OrderedDict(list_layers))
    model = OurModelHacking(backbone, is_B_trainable=is_B_trainable, B_init=None, device=args.device)
    model = model.to(args.device)
    print('Number of trainable var: %d\n' % aux_tools.count_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(
    #         [{'params': model._B, 'lr': 5e-1},
    #          {'params': model.cluster_projector.parameters(), 'lr': 1e-1},#1e-1
    #          ], lr=1e-3)

    code_name = f'our_model-ds={dataset_name}-m={m}-noise={str(p):s}-trial={trial:d}-L={len(list_hiddens)}'
    if is_B_trainable:
        code_name = f'{code_name}-lam={lam:e}'
    model_path = 'save/%s'%code_name
    print(f'Save path: {model_path}')
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = None #SummaryWriter(os.path.join('.runs', code_name+'_'+current_time))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # train
    train(model, optimizer, data_loader, writer, model_path, lam, dataset.X.to(args.device), args)

    END = time.time()
    print('elapsed time: %f s' %  (END - START))


def main(list_m, dataset_name):
    num_trials = 10
    lr = 1e-3
    epochs = 300
    list_hiddens = [3, 128, 3]
    list_lam = [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # list_lam = [0.0, 1e-2]
    p = 0.1

    with Parallel(n_jobs=10) as parallel:
        for m in list_m:
            is_B_trainable = False
            parallel(delayed(sub_main)(0.0, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs, lr) for trial in range(num_trials))

            for lam in list_lam:
                is_B_trainable = True
                parallel(delayed(sub_main)(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs, lr) for trial in range(num_trials))



