import torch
import numpy as np
import pickle as pkl

from our_model_eval_utils import direct_inference, inference, sub_main, main

np.set_printoptions(suppress=True, precision=4)



if __name__ == "__main__":
    m = 10000
    dataset_name = 'cifar10-byol-15k'

    print(f'dataset: {dataset_name}')
    p='e'
    trial = 0
    list_hiddens = [512, 2048, 10]
    epochs = 100
    lam = 1e-3
    is_B_trainable = True

    with open('datasets/cifar10/byol/features.p', 'rb') as i_f:
        data_pkl = pkl.load(i_f)
    X_train = data_pkl[0][:15000]
    y_train = data_pkl[1][:15000]
    X_val = data_pkl[0][15000:20000]
    y_val = data_pkl[1][15000:20000]
    X_test = np.concatenate((data_pkl[2], data_pkl[0][20000:]))
    y_test = np.concatenate((data_pkl[3], data_pkl[1][20000:]))

    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    X_val = torch.from_numpy(X_val.astype(np.float32))
    y_val = torch.from_numpy(y_val.astype(np.float32))
    sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, epochs, interval=5, debug=True)

