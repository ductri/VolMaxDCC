import os
import argparse
import torch
import torchvision
import numpy as np
from joblib import Parallel, delayed
import pickle as pkl

from our_model import OurModel
from my_dataset import StandardDataset
from evaluation import unified_metrics
from utils import aux_tools


np.set_printoptions(suppress=True, precision=4)


def direct_inference(x, y, model, device):
    model.eval()
    predictions = []
    labels_vector = []
    x = x.to(device)
    with torch.no_grad():
        c = model.forward_cluster(x)
    c = c.detach()
    predictions.extend(c.cpu().detach().numpy())
    labels_vector.extend(y.numpy())
    predictions = np.array(predictions)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(predictions.shape))
    return predictions, labels_vector

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


def sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, epochs, interval=5, debug=False, num_workers=0):
    device = torch.device("cuda")
    train_dataset = StandardDataset(X_train, y_train)
    eval_dataset = StandardDataset(X_val, y_val)
    test_dataset = StandardDataset(X_test, y_test)
    if debug:
        print(f'train: {len(train_dataset)}, val: {len(eval_dataset)}, test: {len(test_dataset)}')

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        )
    data_loader_eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        )
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        )
    class_num=10


    train_results = []
    eval_results = []
    test_results = []

    for epoch in range(0, epochs, interval):
        if debug:
            print(f'Epoch: {epoch}')
        # initialize model
        model = OurModel(list_hiddens, is_B_trainable)
        B_init = model._get_B().detach().cpu().numpy()

        code_name = f'our_model-ds={dataset_name}-m={m}-noise={p:s}-trial={trial:d}-L={len(list_hiddens)}'
        if is_B_trainable:
            code_name = f'{code_name}-lam={lam:e}'

        trained_model_path = os.path.join('save/%s' % (code_name), 'checkpoint_{}.tar'.format (epoch))
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['net'])
        if debug:
            print(f'Loaded model from {trained_model_path}')
        model = model.to('cuda')

        label_pred, label_true, M_pred = inference(data_loader_train, model, device)
        label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        train_results.append((nmi, ari, acc))
        if debug:
            print(f'Train NMI: {nmi} ARI: {ari} ACC: {acc}')

            if epoch == epochs-interval:
                __import__('pdb').set_trace()

        label_pred, label_true, M_pred = inference(data_loader_eval, model, device)
        label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        eval_results.append((nmi, ari, acc))
        if debug:
            print(f'Eval NMI: {nmi} ARI: {ari} ACC: {acc}')

        label_pred, label_true, M_pred = inference(data_loader_test, model, device)
        label_pred = unified_metrics.match_it_label(label_true, label_pred, 10)
        nmi, ari, acc = unified_metrics.evaluate(label_true, label_pred)
        test_results.append((nmi, ari, acc))
        if debug:
            print(f'Test NMI: {nmi} ARI: {ari} ACC: {acc}')
            print()

    ind = np.argmax([acc for (_, _, acc) in eval_results])
    if debug:
        print(f'best epoch: {ind*interval} with (train, test) = ({train_results[ind]}, {test_results[ind]})')
    return (train_results[ind], test_results[ind], ind, eval_results[ind])


def main(m, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, list_lam, list_list_hiddens, list_p, num_trials, epochs):
    print('m=%d' % m)
    print('#######################')
    print('NO REG ')
    print('#######################')
    noreg_results = []
    is_B_trainable = False
    with Parallel(n_jobs=5) as parallel:
        for p in list_p:
            p_results = []
            print('############ p=%s ############' % p)
            for list_hiddens in list_list_hiddens:
                results = parallel(delayed(sub_main)(0.0, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, epochs) for trial in range(num_trials))
                all_trials = [(train_result[2], test_result[2], train_result[1], test_result[1], train_result[0], test_result[0]) for (train_result,test_result,_,_) in results]
                p_results.append(all_trials)
            noreg_results.append(p_results)


                # means = all_trials.mean(0)
                # stds = all_trials.std(0)
                # print(f'L={len(list_hiddens)} (acc, nmi, ari) = ({means[0]:.4f}±{stds[0]:.4f}, {means[1]:.4f}±{stds[1]:.4f} & {means[2]:.4f}±{stds[2]:.4f}, {means[3]:.4f}±{stds[3]:.4f} & {means[4]:.4f}±{stds[4]:.4f}, {means[5]:.4f}±{stds[5]:.4f})')
                # print()


    print('#######################')
    print('REG ')
    print('#######################')
    reg_results = []
    is_B_trainable = True
    with Parallel(n_jobs=6) as parallel:
        for p in list_p:
            p_results = []
            print('############ p=%s ############' % p)
            for list_hiddens in list_list_hiddens:
                h_results = []
                for trial in range(num_trials):
                    results = parallel(delayed(sub_main)(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, epochs) for lam in list_lam)
                    best_lam_ind = np.argmax([eval_result[2] for (_, _, _, eval_result) in results])
                    print(f'Trial {trial} has the best lam={list_lam[best_lam_ind]}')
                    train_result, test_result, _, _ = results[best_lam_ind]
                    h_results.append((train_result[2], test_result[2], train_result[1], test_result[1], train_result[0], test_result[0]))
                p_results.append(h_results)
            reg_results.append(p_results)

                # means = all_trials.mean(0)
                # stds = all_trials.std(0)
                # print(f'L={len(list_hiddens)} (acc, nmi, ari) = ({means[0]:.4f}±{stds[0]:.4f}, {means[1]:.4f}±{stds[1]:.4f} & {means[2]:.4f}±{stds[2]:.4f}, {means[3]:.4f}±{stds[3]:.4f} & {means[4]:.4f}±{stds[4]:.4f}, {means[5]:.4f}±{stds[5]:.4f})')
                # print(f'L={len(list_hiddens)} (acc, nmi, ari) = ({means[0]:.2f}, {means[1]:.2f} & {means[2]:.2f}, {means[3]:.2f} & {means[4]:.2f}, {means[5]:.2f})')
                # print()


    reg_results = np.array(reg_results)
    noreg_results = np.array(noreg_results)
    path_to_file = f'results/{dataset_name}_{m}.pkl'
    with open(path_to_file, 'wb') as o_f:
        pkl.dump({'reg_results': reg_results, 'noreg_results': noreg_results}, o_f)

    print(f'Saved results to {path_to_file}')

