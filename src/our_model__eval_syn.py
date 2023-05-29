import os
import argparse
import torch
import torchvision
import numpy as np
from joblib import Parallel, delayed
import pickle
import matplotlib.pyplot as plt

from our_model import OurModel
from my_dataset import StandardDataset
from evaluation import unified_metrics
from utils import aux_tools


np.set_printoptions(suppress=True, precision=4)


def inference(loader, model, device):
    model.eval()
    predictions = []
    labels_vector = []
    probs = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_single(x)
            probs.append(c.cpu().numpy())
        c = c.argmax(dim=-1)
        predictions.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
    predictions = np.array(predictions)
    labels_vector = np.array(labels_vector)
    probs = np.concatenate(probs)
    # print("Features shape {}".format(predictions.shape))
    return predictions, labels_vector, probs


def sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs, B_init=None):
    device = torch.device("cuda")

    with open('datasets/synthetic/data_noise_%s_m_%d_trial_%d.pkl' % (str(p), m, trial), 'rb') as input_file:
        data = pickle.load(input_file)
    A = data['A']
    X = torch.from_numpy(data['X'].T)
    M = torch.from_numpy(data['M'])
    M_true_train = M.cpu().numpy()
    P = (M.T@A.T@A@M).cpu().numpy()
    train_dataset = StandardDataset(X, M.T)

    with open('datasets/synthetic/data_noise_%s_m_%d_trial_%d_test.pkl' % (str(p), m, trial), 'rb') as input_file:
        data = pickle.load(input_file)
    X = torch.from_numpy(data['X'].T)
    M = torch.from_numpy(data['M'])
    M_true_test = M.cpu().numpy()
    test_dataset = StandardDataset(X, M.T)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        )
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        )
    class_num=3


    train_results = []
    test_results = []

    interval = 25
    for epoch in range(0, epochs+1, interval):
        # initialize model
        with torch.no_grad():
            model = OurModel(list_hiddens, is_B_trainable, B_init=None)
            B_init = model._get_B().detach().cpu().numpy()

            code_name = f'our_model-ds={dataset_name}-m={m}-noise={str(p):s}-trial={trial:d}-L={len(list_hiddens)}'
            if is_B_trainable:
                code_name = f'{code_name}-lam={lam:e}'

            trained_model_path = os.path.join('save/%s' % (code_name), 'checkpoint_{}.tar'.format (epoch))
            print(f'Load checkpoint at {trained_model_path}')
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')

            _, _, M_pred = inference(data_loader_train, model, device)
            M_pred = M_pred.T
            P_pred = M_pred.T@model._get_B().cpu().numpy()@M_pred
            print(f'P err: {((P - P_pred)**2).sum()/(P**2).sum():e}')
            mse_train = unified_metrics.MSE(M_true_train, M_pred)
            mse_train = mse_train.item()
            train_results.append(mse_train)
            print(f'Train({len(train_dataset)})- Epoch: {epoch} - Rel error: {mse_train:e}')

            new_M_true, _, _, new_M_pred = unified_metrics.match_it(M_pred, M_true_train)
            print(f'M err: {((new_M_true - new_M_pred)**2).sum()/(M_true_train**2).sum():e}')
            print()

            _, _, M_pred = inference(data_loader_test, model, device)
            M_pred = M_pred.T
            mse_test = unified_metrics.MSE(M_true_test, M_pred)
            mse_test = mse_test.item()
            test_results.append(mse_test)
            print(f'Test({len(test_dataset)}) - Epoch: {epoch} - Rel error: {mse_test:e}')

    best_ind = np.argmin(train_results)
    print(f'Best epoch {best_ind*interval} with train={train_results[best_ind]:e}')
    return train_results[best_ind], test_results[best_ind]

        # # print('EVAl')
        # label_pred, label_true, M_pred = inference(data_loader_eval, model, device)
        # label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        # nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        # eval_results.append((nmi, ari, acc))
        # # print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')
        #
        # # print('TEST')
        # label_pred, label_true, M_pred = inference(data_loader_test, model, device)
        # label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        # nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        # test_results.append((nmi, ari, acc))
        # # print(f'NMI: {nmi} ARI: {ari} ACC: {acc}')

    # ind = np.argmax([acc for (_, _, acc) in eval_results])
    # print(f'best epoch: {ind*5}')
    # print(train_results)
    # return (train_results[ind], test_results[ind], ind, eval_results[ind])


def main(list_m, dataset_name):
    list_hiddens = [3, 128, 3]
    num_trials = 10
    epochs = 300
    p = 0.1

    is_B_trainable = True

    all_results = []
    list_lam = [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # list_lam = [0.0, 1e-2]
    for m in list_m:
        # results = []
        # for lam in list_lam:
        #     with Parallel(n_jobs=10) as parallel:
        #         results.append(parallel(delayed(sub_main)(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs) for trial in range(num_trials)))
        # all_results.append(results)

        with Parallel(n_jobs=10) as parallel:
            all_results.append(parallel(delayed(sub_main)(0.0, False, p, trial, m, list_hiddens, dataset_name, epochs) for trial in range(num_trials)))

    all_results = np.array(all_results)
    return all_results



if __name__ == "__main__":
    list_m = list(range(1000, 10001, 1000))
    # list_m = [1000, 10000]
    dataset_name = 'syn1-noise'
    print(f'dataset: {dataset_name}')
    results = main(list_m, dataset_name)
    with open('syn_results/syn_noise_result_without_B.pkl', 'wb') as output_file:
        pickle.dump(results, output_file)


    # list_hiddens = [3, 128, 3]
    # epochs = 300
    # trial = 0
    # is_B_trainable = True
    # m = 1000
    # dataset_name = 'syn1-noise'
    # p = 0.1
    # lam = 1e-1
    # sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, epochs)
    #
