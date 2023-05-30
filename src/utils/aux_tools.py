import random
import os
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle as pkl
import pandas as pd

import stl10_oracle__train
import cifar10_oracle__train
from imagenet10_oracle__train import ImageNet10Oracle
import oracle_cifar10

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(name, batch_size=32, num_workers=8, p=0.001, include_dataset=False, e=0.01, m=-1,trial=-1):
    if name[:7]=='cifar10':
        data_pairs = load_cifar10(m, p, trial, name)
    elif name[:14] == 'imagenet10-raw':
        data_pairs = load_dataset(m, p, trial, 'imagenet10', name)
    elif name[:11]=='sub-cifar10':
        data_pairs = load_cifar10_sub(m, p, trial, name, 10000)
    elif name[:5]=='stl10':
        data_pairs = load_stl10(m, p, trial, name)
    elif name=='syn1':
        data_pairs = load_syn1(m, p, trial)
    elif name=='syn1-noise':
        data_pairs = load_syn1_noise(m, p, trial)
    else:
        raise Exception('typo')

    data_loader = torch.utils.data.DataLoader(
        data_pairs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    if include_dataset:
        return data_loader, data_pairs
    else:
        return data_loader

class PairDataset(Dataset):
    def __init__(self, ind_pairs, label_pairs, X, others=[]):
        self.ind_pairs = ind_pairs
        self.label_pairs = label_pairs
        self.X = torch.from_numpy(X)
        self.others = others

    def __len__(self):
        return len(self.ind_pairs)

    def __getitem__(self, idx):
        i1, i2 = self.ind_pairs[idx]
        return (self.X[i1], self.X[i2]), self.label_pairs[idx]

    def getitem_extra(self, idx):
        i1, i2 = self.ind_pairs[idx]
        return (self.X[i1], self.X[i2]), self.label_pairs[idx], (i1, i2, self.others[idx])

class PairDatasetRaw(Dataset):
    def __init__(self, dataset, ind_pairs, shuffled_inds):
        self.dataset = dataset
        self.ind_pairs = ind_pairs
        self.shuffled_inds = shuffled_inds

    def __len__(self):
        return len(self.ind_pairs)

    def __getitem__(self, idx):
        i1, i2 = self.ind_pairs[idx]
        i1 = self.shuffled_inds[i1]
        i2 = self.shuffled_inds[i2]

        X1 = self.dataset[i1][0]
        y1 = self.dataset[i1][1]
        X2 = self.dataset[i2][0]
        y2 = self.dataset[i2][1]
        return (X1, X2), np.float32(1.0*(y1==y2))

def freeze_it(model):
    for param in model.parameters():
        param.requires_grad = False


def load_stl10(m, p, trial, codename):
    pair_path = 'datasets/stl10/pair_%d_real_%d_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        true_label_pairs = data_dict['true_label_pairs']
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()

        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
    else:
        print('Loading raw dataset')
        if codename == 'stl10-real':
            X = np.load('datasets/stl10/feature.npy')[:10000]
            X = torch.from_numpy(X.astype(np.float32))
            true_y = np.load('datasets/stl10/label.npy')[:10000]
        elif codename == 'stl10-simclr_pytorch2':
            with open('datasets/stl10/simclr_pytorch2/stl10.pkl', 'rb') as i_f:
                data_pkl = pkl.load(i_f)
            X = data_pkl['train_X']
            X = torch.from_numpy(X.astype(np.float32))
            true_y = data_pkl['train_y']
        else:
            raise Exception(f'Codename {codename} does not exist!')

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        i1 = np.random.randint(0, N, size=m).tolist()
        random.shuffle(i1)
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y=[]
        if p!=-1:
            model = stl10_oracle__train.STL10Oracle(512, 10)
            trained_model_path = os.path.join('save/stl10-oracle/', "checkpoint_{:d}.tar".format(p))
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        # ind_pairs, label_pairs = make_pairs_random_2(y, m)
        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')
    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset


def load_cifar10(m, p, trial, codename):
    pair_path = f'datasets/cifar10/pairs/pair_{m:d}_{str(p):s}_{codename}_trial_{trial}.pkl'
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = pkl.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        true_y = data_dict['true_y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        if codename == 'cifar10-real':
            X = np.load('datasets/cifar10/feature_train.npy')[:10000]
            X = torch.from_numpy(X.astype(np.float32))
            true_y = np.load('datasets/cifar10/label_train.npy')[:10000]
        elif codename == 'cifar10-simclr_pytorch':
            with open('datasets/cifar10/simclr_pytorch/cifar10_train.pkl', 'rb') as i_f:
                data_pkl = pkl.load(i_f)
            X = data_pkl['X']
            X = torch.from_numpy(X.astype(np.float32))
            true_y = data_pkl['Y']
        elif codename == 'cifar10-simclr_pytorch-15k':
            with open('datasets/cifar10/simclr_pytorch/cifar10_train.pkl', 'rb') as i_f:
                data_pkl = pkl.load(i_f)
            X = data_pkl['X'][:15000]
            X = X.astype(np.float32)
            true_y = data_pkl['Y'][:15000]
        elif codename == 'cifar10-byol-15k':
            with open('datasets/cifar10/byol/features.p', 'rb') as i_f:
                data_pkl = pkl.load(i_f)
            X = data_pkl[0][:15000]
            true_y = data_pkl[1][:15000]
        else:
            raise Exception(f'Codename {codename} does not exist!')

        base_pair_path = f'datasets/cifar10/pairs/pair_{m:d}_-1_{codename}_trial_{trial}.pkl'
        if os.path.exists(base_pair_path):
            with open(base_pair_path, 'rb') as i_f:
                base_data_dict = pkl.load(i_f)
            ind_pairs = base_data_dict['ind_pairs']
            i1, i2 = zip(*ind_pairs)
            i1 = np.array(i1)
            i2 = np.array(i2)
        else:
            print('Creating pair dataset')
            N = X.shape[0]
            K = 10
            i1 = np.random.randint(0, N, size=m)
            i2 = np.random.randint(0, N, size=m)
            ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        if str(p)!='-1':
            noise_dict = {'a': '100', 'b': '50', 'c': '10', 'd': '5', 'e': '4'}
            model = oracle_cifar10.models.ResNet18()
            model = model.to('cuda')
            trained_model_path = os.path.join(f'oracle_checkpoint/cifar10/', f'ckpt_{noise_dict[p]}.pth')
            checkpoint = torch.load(trained_model_path)
            print('Reloaded the saved model from %s \n' % trained_model_path)
            my_checkpoint = OrderedDict()
            for key, value in checkpoint['net'].items():
                if key[:6] == 'module':
                    my_checkpoint[key[7:]] = checkpoint['net'][key]
                else:
                    my_checkpoint[key] = checkpoint['net'][key]
            model.load_state_dict(my_checkpoint)

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = torchvision.datasets.CIFAR10(
                root='./datasets/cifar10', train=True, download=True, transform=transform_test)
            if codename == 'cifar10-simclr_pytorch-15k' or codename == 'cifar10-byol-15k':
                trainset = Subset(trainset, range(15000))
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=False, num_workers=2)

            y_pred = []
            true_y_2 = []
            with torch.no_grad():
                for batch in tqdm(train_loader):
                    pred = model(batch[0].to('cuda'))
                    y_pred.append(pred.argmax(1).cpu().numpy())
                    true_y_2.append(batch[1].cpu().numpy())
            y_pred = np.concatenate(y_pred)
            true_y_2 = np.concatenate(true_y_2)
            assert np.all(true_y_2 == true_y)
            label_pairs = (y_pred[i1]==y_pred[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            pkl.dump({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'true_y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [true_y])
    return pair_dataset

def load_dataset(m, p, trial, dataset_name, codename):
    pair_path = f'datasets/{dataset_name}/pairs/pair_{m:d}_{str(p):s}_{codename}_trial_{trial}.pkl'
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = pkl.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        label_pairs = label_pairs.astype(np.float32)
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        true_y = data_dict['true_y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        if codename == 'imagenet10-cc-10k':
            with open('datasets/imagenet10/imagenet10.pkl', 'rb') as i_f:
                data_dict = pkl.load(i_f)
            with open('datasets/imagenet10/real/meta_data.pkl', 'rb') as i_f:
                meta_data_dict = pkl.load(i_f)

            N = meta_data_dict['N']
            shuffled_inds = meta_data_dict['shuffle_inds'][:N]
            X = data_dict['X'][shuffled_inds]
            X_path = np.array(data_dict['path'])[shuffled_inds]
            true_y = data_dict['Y'][shuffled_inds]

        else:
            raise Exception(f'Codename {codename} does not exist!')

        base_pair_path = f'datasets/{dataset_name}/pairs/pair_{m:d}_real_{codename}_trial_{trial}.pkl'
        if p == 'real':
            i1 = meta_data_dict['i1'][:m]
            i2 = meta_data_dict['i2'][:m]
            ind_pairs = list(zip(i1, i2))
            print('Loaded pair of indices from an external file!')
        else:
            if os.path.exists(base_pair_path):
                with open(base_pair_path, 'rb') as i_f:
                    base_data_dict = pkl.load(i_f)
                ind_pairs = base_data_dict['ind_pairs']
                i1, i2 = zip(*ind_pairs)
                i1 = np.array(i1)
                i2 = np.array(i2)
                print(f'Loaded pair of indices from {base_pair_path}')
            else:
                print('Creating new pair indices')
                N = X.shape[0]
                K = 10
                i1 = np.random.randint(0, N, size=m)
                i2 = np.random.randint(0, N, size=m)
                ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        if p == 'real':
            df = pd.read_csv('datasets/imagenet10/real/pair_labels/trial_9.csv')
            real_pairwise_labels_dict = {}
            for i in range(df.shape[0]):
                url = df.loc[i, 'Input.image_url']
                start = url.index('.com')+5
                end = url[start:].index('datasets')-1
                pair_inds = url[start:start+end].split('-')
                ind1, ind2 = int(pair_inds[0]), int(pair_inds[1])
                real_pairwise_labels_dict[(ind1, ind2)] = (df.loc[i, 'Answer.category.label'].lower() == 'yes')*1.0
            #TODO just for fun
            label_pairs = np.array([real_pairwise_labels_dict[(x, y)] if (x, y) in real_pairwise_labels_dict else 0 for x,y in zip(i1, i2)])
            label_pairs = label_pairs.astype(np.float32)
            print('Label of pairs are loaded from an external file')
        elif str(p) !='-1':
            noise_dict = {'a': '100', 'b': '50', 'c': '10', 'd': '5', 'e': '4'}
            if dataset_name == 'imagenet10':
                model = oracle_cifar10.models.ResNet18()

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                trainset = torchvision.datasets.CIFAR10(
                    root='./datasets/cifar10', train=True, download=True, transform=transform_test)
                if codename == 'cifar10-simclr_pytorch-15k':
                    trainset = Subset(trainset, range(15000))
                train_loader = torch.utils.data.DataLoader(
                    trainset, batch_size=128, shuffle=False, num_workers=2)
            else:
                raise Exception(f'dataset name {dataset_name} does not exist.')

            model = model.to('cuda')
            trained_model_path = os.path.join(f'oracle_checkpoint/{dataset_name}/', f'ckpt_{noise_dict[p]}.pth')
            checkpoint = torch.load(trained_model_path)
            print('Reloaded the saved model from %s \n' % trained_model_path)
            my_checkpoint = OrderedDict()
            for key, value in checkpoint['net'].items():
                if key[:6] == 'module':
                    my_checkpoint[key[7:]] = checkpoint['net'][key]
                else:
                    my_checkpoint[key] = checkpoint['net'][key]
            model.load_state_dict(my_checkpoint)


            y_pred = []
            true_y_2 = []
            with torch.no_grad():
                for batch in tqdm(train_loader):
                    pred = model(batch[0].to('cuda'))
                    y_pred.append(pred.argmax(1).cpu().numpy())
                    true_y_2.append(batch[1].cpu().numpy())
            y_pred = np.concatenate(y_pred)
            true_y_2 = np.concatenate(true_y_2)
            assert np.all(true_y_2 == true_y)
            label_pairs = (y_pred[i1]==y_pred[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
            print('Label of pairs has been created via a classifer')
        else:
            label_pairs = true_label_pairs.astype(np.float32)
            print('Label of pairs has been constructed using true class labels')

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            pkl.dump({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'true_y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [true_y])
    return pair_dataset

def load_imagenet10_real(m, p, trial):
    pair_path = 'datasets/imagenet-10/pair_%d_real_%d_trial_%d.pt' % (m, p, trial)
    if os.path.exists(pair_path):
        print('Load pair dataset from %s' % pair_path)
        with open(pair_path, 'rb') as file_handler:
            data_dict = torch.load(file_handler)
        ind_pairs = data_dict['ind_pairs']
        label_pairs = data_dict['label_pairs']
        true_label_pairs = data_dict['true_label_pairs']
        X = data_dict['X']
        y = data_dict['y']
        print('#Nodes: %d \t #Edges: %d \n' % (len(X), len(ind_pairs)))
        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()
    else:
        print('Loading raw dataset')
        X = np.load('datasets/imagenet-10/feature.npy')[:10000]
        X = torch.from_numpy(X.astype(np.float32))
        true_y = np.load('datasets/imagenet-10/label.npy')[:10000]

        print('Creating pair dataset')
        N = X.shape[0]
        K = 10
        if m>N:
            i1 = list(range(N)) + np.random.randint(0, N, size=(m-N)).tolist()
        else:
            i1 = np.random.randint(0, N, size=m).tolist()
        random.shuffle(i1)
        i1 = np.array(i1)
        i2 = np.random.randint(0, N, size=m)
        ind_pairs = list(zip(i1, i2))
        true_label_pairs = (true_y[i1] == true_y[i2])*1.0

        y= []
        if p!=-1:
            model = ImageNet10Oracle(512, 10)
            trained_model_path = os.path.join('save/imagenet10-oracle/', "checkpoint_{:d}.tar".format(p)) #[2, 3, 9]
            print('Reload saved model from %s \n' % trained_model_path)
            checkpoint = torch.load(trained_model_path)
            model.load_state_dict(checkpoint['net'])
            model = model.to('cuda')
            with torch.no_grad():
                y = model(X.to('cuda'))
                y = y.argmax(1).cpu().numpy()
            label_pairs = (y[i1]==y[i2])*1.0
            label_pairs = label_pairs.astype(np.float32)
        else:
            label_pairs = true_label_pairs.astype(np.float32)

        flipping_rate = 1-((label_pairs==true_label_pairs)*1.0).mean()


        # ind_pairs, label_pairs = make_pairs_random_2(y, m)
        print('#Nodes: %d \t #Edges: %d \n' % (X.shape[0], len(label_pairs)))
        with open(pair_path, 'wb') as file_handler:
            torch.save({'ind_pairs': ind_pairs, 'label_pairs': label_pairs, \
                    'X': X, 'y': true_y, 'true_label_pairs':true_label_pairs}, file_handler)
        print('Save to %s' % pair_path)
    print(f'Flipping rate: {flipping_rate:.4}')

    pair_dataset = PairDataset(ind_pairs, label_pairs, X, [y])
    return pair_dataset

def save_model2(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)
    print('Save model to %s' % model_path)

