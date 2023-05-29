import torch
import numpy as np
import pickle as pkl

from our_model_eval_utils import direct_inference, inference, sub_main, main

np.set_printoptions(suppress=True, precision=4)



if __name__ == "__main__":
    m = 10002
    dataset_name = 'stl10-simclr_pytorch2'
    print(f'dataset: {dataset_name}')

    p='-1'
    trial = 0
    list_hiddens = [2048, 10]

    with open('datasets/stl10/simclr_pytorch2/stl10.pkl', 'rb') as i_f:
        data_pkl = pkl.load(i_f)
        X_train = data_pkl['train_X']
        y_train = data_pkl['train_y']
        X_test = data_pkl['test_X']
        y_test = data_pkl['test_y']

    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    epochs = 100
    lam = 0.0
    sub_main(lam, False, p, trial, m, list_hiddens, dataset_name, X_train, y_train, X_test, y_test, epochs, eval_size=1000, interval=5)

