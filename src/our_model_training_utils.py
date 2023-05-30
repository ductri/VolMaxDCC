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
from our_model import OurModel


def train(model, optimizer, data_loader, model_path, lam, X, args):
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epochs):
        loss_epoch = 0
        loss1_epoch = 0
        loss2_epoch = 0
        acc_epoch = 0
        trange = tqdm(enumerate(data_loader), total=len(data_loader))
        total_norm = 0.0
        for step, ((x_i, x_j), y) in trange:
            optimizer.zero_grad()
            x_i = x_i.to('cuda')
            x_j = x_j.to('cuda')
            y = y.to('cuda')
            c_i, c_j = model.forward(x_i, x_j)
            loss, loss1, loss2 = model.compute_loss(model._get_B(), c_i, c_j, y, lam, X)

            loss.backward()

            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()
            loss_epoch += loss.item()
            loss1_epoch += loss1.item()
            loss2_epoch += loss2.item()
            with torch.no_grad():
                prob_pred = model.link_prediction(model._get_B(), c_i, c_j)
                acc = aux_tools.get_accuracy(prob_pred>=0.5, y)
                acc_epoch += acc


        if epoch % 5 == 0:
            aux_tools.save_model2(model_path, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}] \t Loss: {loss_epoch / len(data_loader):.5} Loss1: {loss1_epoch / len(data_loader):.5} Loss2: {loss2_epoch / len(data_loader):.5} Acc: {acc_epoch / len(data_loader):.5} lam={lam:.5} grad_norm={total_norm/len(data_loader):.5}")
    aux_tools.save_model2(model_path, model, optimizer, args.epochs)

def sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs=100, lr=1e-2, B_init=None, num_workers=0, debug=False, optimizer_name='adam'):
    START = time.time()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs=epochs

    seed = 101
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # prepare data
    data_loader, dataset = aux_tools.load_data(dataset_name, batch_size=128, \
            num_workers=num_workers, m=m, include_dataset=True, p=p, trial=trial)
    if debug:
        print(f'Examples of index pairs: {str(dataset.ind_pairs[:5])}')
    class_num = 10

    model = OurModel([512, 512, 10], is_B_trainable=is_B_trainable, B_init=None, device=device)
    model = model.to('cuda')
    print('Number of trainable var: %d\n' % aux_tools.count_parameters(model))

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
                [{'params': model._B, 'lr': 1e-1},
                 {'params': model.cluster_projector.parameters(), 'lr': lr}, #1e-1
                 ], lr=1e-3)
    else:
        raise Exception('not proper optimizer')

    code_name = f'our_model-ds={dataset_name}-m={m}-noise={str(p):s}-trial={trial:d}-L={len(list_hiddens)}'
    if is_B_trainable:
        code_name = f'{code_name}-lam={lam:e}'
    model_path = 'save/%s'%code_name
    print(f'Save path: {model_path}')
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # train
    # try:
    train(model, optimizer, data_loader, model_path, lam, None, args)
    # except Exception as e:
    #     print('EXCEPTION happened')
    #     print(e)
    #     raise

    END = time.time()
    print('elapsed time: %f s' %  (END - START))


def main(m, dataset_name, list_p, num_trials, lr, epochs, list_list_hiddens):
    list_lam = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    with Parallel(n_jobs=4) as parallel:
        for list_hiddens in list_list_hiddens:
            is_B_trainable = False
            for p in list_p:
                parallel(delayed(sub_main)(0.0, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs, lr) for trial in range(num_trials))

            is_B_trainable = True
            for p in list_p:
                for trial in range(num_trials):
                    parallel(delayed(sub_main)(lam,  is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs, lr) for lam in list_lam)


