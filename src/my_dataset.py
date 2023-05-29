from torch.utils.data import Dataset


class StandardDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, ind):
        return self.X[ind, :], self.y[ind]

class ShuffledDataset(Dataset):
    def __init__(self, X, y, shuffle_inds):
        self.X = X
        self.y = y
        self.shuffle_inds

    def __len__(self):
        return len(self.y)

    def __getitem__(self, ind):
        return self.X[self.shuffle_inds[ind], :], self.y[self.shuffle_inds[ind]]

