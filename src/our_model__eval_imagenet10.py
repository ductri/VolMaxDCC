import os

import torch
import numpy as np
import pickle as pkl

from our_model_eval_utils import direct_inference, inference, sub_main, main

np.set_printoptions(suppress=True, precision=4)



if __name__ == "__main__":
    dataset_name = 'imagenet10-cc-10k'
    print(f'dataset: {dataset_name}')
    p='real'
    trial = 0
    list_hiddens = [512, 512, 10]
    epochs = 100
    lam = 1e-2
    is_B_trainable = True

    with open('datasets/imagenet10/imagenet10.pkl', 'rb') as i_f:
        data_pkl = pkl.load(i_f)
    with open('datasets/imagenet10/real/meta_data.pkl', 'rb') as i_f:
        meta_dict = pkl.load(i_f)
    shuffled_inds = meta_dict['shuffle_inds']

    N = meta_dict['N']
    X_train = data_pkl['X'][shuffled_inds[:N]]
    X_val = data_pkl['X'][shuffled_inds[N:N+1000]]
    X_test = data_pkl['X'][shuffled_inds[N+1000:]]

    y_train = data_pkl['Y'][shuffled_inds[:N]]
    y_val = data_pkl['Y'][shuffled_inds[N:N+1000]]
    y_test = data_pkl['Y'][shuffled_inds[N+1000:]]

    m = 8994
    list_lam = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    list_list_hiddens = [[512, 512, 10],]
    list_p = ['real']
    num_trials = 1
    epochs = 100
    path_to_file = f'results/{dataset_name}_{m}.pkl'
    if not os.path.exists(path_to_file):
        main(m, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, list_lam, list_list_hiddens, list_p, num_trials, epochs)
    with open(path_to_file, 'rb') as i_f:
        results = pkl.load(i_f)
    print('Vanilla')
    result = results['noreg_results']

    # p level
    for i in range(len(list_p)):
        acc = (result[i, 0, :, 0].mean(), result[i, 0, :, 1].mean())
        acc_std = (result[i, 0, :, 0].std(), result[i, 0, :, 1].std())
        nmi = (result[i, 0, :, 2].mean(), result[i, 0, :, 3].mean())
        nmi_std = (result[i, 0, :, 2].std(), result[i, 0, :, 3].std())
        ari = (result[i, 0, :, 4].mean(), result[i, 0, :, 5].mean())
        ari_std = (result[i, 0, :, 4].std(), result[i, 0, :, 5].std())
        print(f'p = {list_p[i]}')
        print(f'(acc, nmi, ari) = \n \
                ({acc[0]:.4f}±{acc_std[0]:.4f}, {acc[1]:.4f}±{acc_std[1]:.4f}),\n \
                ({nmi[0]:.4f}±{nmi_std[0]:.4f}, {nmi[1]:.4f}±{nmi_std[1]:.4f}), \n \
                ({ari[0]:.4f}±{ari_std[0]:.4f}, {ari[1]:.4f}±{ari_std[1]:.4f})')


    result = results['reg_results']
    print('VolMax')
    for i in range(len(list_p)):
        acc = (result[i, 0, :, 0].mean(), result[i, 0, :, 1].mean())
        acc_std = (result[i, 0, :, 0].std(), result[i, 0, :, 1].std())
        nmi = (result[i, 0, :, 2].mean(), result[i, 0, :, 3].mean())
        nmi_std = (result[i, 0, :, 2].std(), result[i, 0, :, 3].std())
        ari = (result[i, 0, :, 4].mean(), result[i, 0, :, 5].mean())
        ari_std = (result[i, 0, :, 4].std(), result[i, 0, :, 5].std())
        print(f'p = {list_p[i]}')
        print(f'(acc, nmi, ari) = \n \
                ({acc[0]:.4f}±{acc_std[0]:.4f}, {acc[1]:.4f}±{acc_std[1]:.4f}),\n \
                ({nmi[0]:.4f}±{nmi_std[0]:.4f}, {nmi[1]:.4f}±{nmi_std[1]:.4f}),\n \
                ({ari[0]:.4f}±{ari_std[0]:.4f}, {ari[1]:.4f}±{ari_std[1]:.4f})')

