import torch
import numpy as np
import pickle as pkl

from our_model_eval_utils import direct_inference, inference, sub_main, main

np.set_printoptions(suppress=True, precision=4)



if __name__ == "__main__":
    m = 8994
    dataset_name = 'imagenet10-cc-10k'

    print(f'dataset: {dataset_name}')
    p='real'
    trial = 0
    list_hiddens = [512, 512, 10]
    epochs = 100
    lam = 1e-2
    is_B_trainable = False

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


    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    X_val = torch.from_numpy(X_val.astype(np.float32))
    y_val = torch.from_numpy(y_val.astype(np.float32))
    sub_main(lam, is_B_trainable, p, trial, m, list_hiddens, dataset_name, X_train, y_train, X_test, y_test, X_val, y_val, epochs, interval=5, debug=True, num_workers=2)

