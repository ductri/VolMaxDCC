import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from scipy.io import savemat
from joblib import Parallel, delayed
np.set_printoptions(suppress=True, precision=2)


def plot_simplex(M):
    """
    M: N x K
    vertices: K x 2
    """
    vertices = np.array([[-0.7, 0], [0.7, 0], [0, 1]])
    new_coor = np.matmul(M.T,vertices)
    plt.figure()
    colors = [(1.0, a, b)  for a, b in zip(M[0, :], M[1, :])]
    plt.scatter(new_coor[:, 0], new_coor[:, 1], marker='x', color=colors)
    plt.title('K = %d' % M.shape[0])
    plt.savefig('./datasets/synthetic/membership.pdf', bbox_inches='tight')

def f_inv(M):
    """
    M: 3 x N
    """
    X = np.zeros(M.shape)
    X[0, :] = 2*M[0, :]
    X[1, :] = M[1, :]**4
    X[2, :] = M[2, :]**2 + 0.5
    # K, N = M.shape
    # X = np.random.randn(K, K)@np.log(M) + np.random.randn(K, 1)
    return X

def generate_data(m, noise_level):
    K = 3
    N = 100

    # M = np.zeros((K, N))
    # M[np.random.randint(0, 3, int(N)), np.arange(int(N))] = 1
    # M = M + 0.02*np.random.rand(K, int(N))
    # M[M<0] = 0
    # M = M/M.sum(0)
    M = np.random.dirichlet(0.2*np.ones(3), N).T

    plot_simplex(M)

    rand_i = np.random.randint(0, N, size=m)
    rand_j = np.random.randint(0, N, size=m)
    # A = np.array(
    #      [[1.0, 0.2, 0.3],
    #      [0.0, 0.8, 0.3],
    #      [0.0, 0.0, 0.4]])
    A = np.eye(3)
    B = A.T@A
    # B = np.array(
    #         [[0.9, 0.5, 0.9],
    #          [0.1, 0.9, 0.5],
    #          [0.4, 0.05, 0.9]])

    P = np.matmul(np.matmul(M.T, B), M)
    y = (np.random.rand(m) <= P[rand_i, rand_j]).astype(np.float32)
    X = f_inv(M).astype(np.float32)
    return rand_i, rand_j, y, X, M, A, N


def sample(m, trial, noise_level):
    print(trial)
    ind_i, ind_j, pair_labels, X, M, A, N = generate_data(m, noise_level)
    data = dict()
    data['i'] = ind_i
    data['j'] = ind_j
    data['pair_labels'] = pair_labels
    data['X'] = X
    data['M'] = M
    data['A'] = A
    data['N'] = N
    data_path = 'datasets/synthetic/data_%s_m_%d_trial_%d.pkl' % (str(noise_level), m, trial)
    with open(data_path,  'wb') as file_handler:
        pickle.dump(data, file_handler)
    print(f'Save to {data_path}')
    B = np.eye(3)
    tmp = (np.matmul(M[:, ind_i].T, B)*M[:, ind_j].T).sum(1)
    loss = -pair_labels*np.log(tmp) - (1-pair_labels)*np.log(1-tmp)
    print(f'MLE={loss.mean()}')





if __name__ == "__main__":

    list_m = [8000]
    # list_m = [200, 500, 800]
    num_trials = 1
    noise_level = 1e-1
    sample(1000, 0, noise_level)
    # with Parallel(n_jobs=10) as parallel:
    #     for m in list_m:
    #         parallel(delayed(sample)(m, trial, noise_level) for trial in range(num_trials))
    #
