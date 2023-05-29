import torch
import pickle as pkl
from tqdm import tqdm


def main():
    p = '-1'
    dataset_name = 'cifar10-simclr_pytorch-15k'
    trial = 0
    for trial in tqdm(list(range(5))):
        for p in ['-1', 'a', 'b', 'c']:
            with open(f'datasets/cifar10/pairs/pair_10000_{p:s}_{dataset_name}_trial_{trial}.pt', 'rb') as i_f:
                data_dict = torch.load(i_f)
            new_data_dict = {}
            new_data_dict['ind_pairs'] = data_dict['ind_pairs']
            new_data_dict['label_pairs'] = data_dict['label_pairs']
            new_data_dict['X'] = data_dict['X'].numpy()
            new_data_dict['y'] = data_dict['y']
            new_data_dict['true_label_pairs'] = data_dict['true_label_pairs']
            with open(f'datasets/cifar10/pairs/pair_10000_{p:s}_{dataset_name}_trial_{trial}.pkl', 'wb') as o_f:
                pkl.dump(new_data_dict, o_f)



if __name__ == "__main__":
    main()
