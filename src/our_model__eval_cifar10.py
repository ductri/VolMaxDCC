import torch
import numpy as np
import pickle as pkl

from our_model_eval_utils import direct_inference, inference, sub_main, main

np.set_printoptions(suppress=True, precision=4)



if __name__ == "__main__":
    # dataset_name = 'cifar10-simclr_pytorch-15k'
    dataset_name = 'cifar10-byol-15k'
    if dataset_name == 'cifar10-simcrl_pytorch-15k':
        with open('./datasets/cifar10/simclr_pytorch/cifar10_train.pkl', 'rb') as i_f:
            data_train = pkl.load(i_f)
        X_train = data_train['X'][:15000]
        y_train = data_train['Y'][:15000]
        X_val = data_train['X'][15000:20000]
        y_val = data_train['Y'][15000:20000]
        with open('./datasets/cifar10/simclr_pytorch/cifar10_test.pkl', 'rb') as i_f:
            data = pkl.load(i_f)
        X_test = np.concatenate((data['X'], data_train['X'][20000:]))
        y_test = np.concatenate((data['Y'], data_train['Y'][20000:]))
    elif dataset_name == 'cifar10-byol-15k':
        with open('datasets/cifar10/byol/features.p', 'rb') as i_f:
            data_pkl = pkl.load(i_f)
        X_train = data_pkl[0][:15000]
        y_train = data_pkl[1][:15000]
        X_val = data_pkl[0][15000:20000]
        y_val = data_pkl[1][15000:20000]
        X_test = np.concatenate((data_pkl[2], data_pkl[0][20000:]))
        y_test = np.concatenate((data_pkl[3], data_pkl[1][20000:]))

    m = 10000
    list_lam = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    list_list_hiddens = [[512, 512, 10],]
    list_p = ['-1', 'b', 'd', 'e']
    num_trials = 5
    epochs = 200
    # main(m, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, list_lam, list_list_hiddens, list_p, num_trials, epochs)
    path_to_file = f'results/{dataset_name}_{m}.pkl'
    with open(path_to_file, 'rb') as i_f:
        results = pkl.load(i_f)
    print('Vanilla')
    result = results['noreg_results']

    # p level
    for i in range(4):
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
    for i in range(4):
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

