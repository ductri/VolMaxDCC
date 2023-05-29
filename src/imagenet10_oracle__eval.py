import os
import argparse
import torch
import torchvision
import numpy as np
np.set_printoptions(suppress=True, precision=4)
from evaluation import evaluation

from imagenet10_oracle__train import ImageNet10Oracle
from my_dataset import StandardDataset
from joblib import Parallel, delayed


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
    print("Features shape {}".format(predictions.shape))
    return predictions, labels_vector

def inference(loader, model, device):
    model.eval()
    predictions = []
    labels_vector = []
    probs = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model(x)
            probs.append(c.cpu().numpy())
        c = c.argmax(dim=-1)
        predictions.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    predictions = np.array(predictions)
    labels_vector = np.array(labels_vector)
    probs = np.concatenate(probs)
    print("Features shape {}".format(predictions.shape))
    return predictions, labels_vector, probs


def sub_main():
    device = torch.device("cuda")

    X = np.load('datasets/imagenet-10/feature_600.npy')
    y = np.load('datasets/imagenet-10/label_600.npy')

    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    train_dataset = StandardDataset(X[:10000], y[:10000])
    test_dataset = StandardDataset(X[10000:], y[10000:])

    data_loader = torch.utils.data.DataLoader(
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
    class_num=10



    train_results = []
    test_results = []

    for epoch in [0, 1, 2, 3]: #range(0, 10, 1): #70
        # initialize model
        model = ImageNet10Oracle(512, class_num)
        trained_model_path = os.path.join('save/imagenet10-oracle-e600/', "checkpoint_{}.tar".format(epoch))
        print('Reload saved model from %s \n' % trained_model_path)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['net'])
        model = model.to('cuda')

        print("### Train ###")
        X, Y, M_pred = inference(data_loader, model, device)
        nmi, ari, f, acc = evaluation.tevaluate(Y, X, class_num)
        print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
        train_results.append(acc)

        print("### Test ###")
        X, Y, _ = inference(data_loader_test, model, device)
        nmi, ari, f, acc = evaluation.tevaluate(Y, X, class_num)
        print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
        test_results.append(acc)

    ind = np.argmax(test_results)
    __import__('pdb').set_trace()
    return (train_results[ind], test_results[ind], ind)


def main():
    sub_main()


if __name__ == "__main__":
    main()
