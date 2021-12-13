import random
import h5py
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
 
def data_load(path_1x, path_high, low_shape, high_shape, random_seed):
    Inputs = []
    Targets = []
    MM = []

    with open(path_1x, 'r') as csv_file_1x:
        with open(path_high, 'r') as csv_file_high:
            reader1 = csv.reader(csv_file_1x)
            reader2 = csv.reader(csv_file_high)
            num = 0
            for row1, row2 in zip(reader1, reader2):
                if num <= 4198: # the total number of samples you want to use
                    arr_1x = np.array([float(i) for i in row1])
                    arr_high = np.array([float(i) for i in row2])

                    input_ = arr_1x.reshape([low_shape, low_shape])[np.newaxis, :, :]
                    target_ = arr_high.reshape([high_shape, high_shape])[np.newaxis, :, :]

                    Inputs.append(input_)
                    Targets.append(target_)
                num += 1
                if num % 100 == 0:
                    print(num)

    lr_min = np.min(Inputs)
    lr_max = np.max(Inputs)
    min_ = lr_min * 7
    max_ = lr_max * 7
    MM = [lr_min, lr_max, min_, max_]  # normalized scale
    print(MM)

    Inputs = (Inputs - min_) / (max_ - min_)
    Targets = (Targets - min_) / (max_ - min_)

    random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)
    state = np.random.get_state()
    np.random.shuffle(Inputs)
    np.random.set_state(state)
    np.random.shuffle(Targets)

    train_num = 3600 # train data number
    print(train_num)

    Inputs_train = np.array(Inputs[:train_num])
    Targets_train = np.array(Targets[:train_num])
    Inputs_test = np.array(Inputs[train_num:])
    Targets_test = np.array(Targets[train_num:])
    # print(Inputs_train.shape, Targets_train.shape)
    # print(Inputs_valid.shape, Targets_valid.shape)
    # print(Inputs_train[0, 0, :])

    print('Inputs training dimensions:', Inputs_train.shape)
    print('Inputs testing dimensions:', Inputs_test.shape)
    print('Targets training dimensions:', Targets_train.shape)
    print('Targets testing dimensions:', Targets_test.shape)
    
    return Inputs_train, Targets_train, Inputs_test, Targets_test, MM


class TrainDataset(Dataset):
    def __init__(self, lr, hr, scale):
        super(TrainDataset, self).__init__()
        self.lr = lr
        self.hr = hr
        self.scale = scale

    @staticmethod
    def random_rotate(lr, hr):
        # rand = random.random()
        if random.random() < 0.25:
            lr = np.rot90(lr, 1, axes=(1, 2)).copy()
            hr = np.rot90(hr, 1, axes=(1, 2)).copy()
        elif random.random() < 0.5:
            lr = np.rot90(lr, -1, axes=(1, 2)).copy()
            hr = np.rot90(hr, -1, axes=(1, 2)).copy()
        elif random.random() < 0.75:
            lr = np.rot90(lr, 2, axes=(1, 2)).copy()
            hr = np.rot90(hr, 2, axes=(1, 2)).copy()
        return lr, hr

    def __getitem__(self, idx):
        lr, hr = self.random_rotate(self.lr[idx, :, :, :], self.hr[idx, :, :, :])
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        return lr, hr

    def __len__(self):
        return self.lr.shape[0]


class EvalDataset(Dataset):
    def __init__(self, lr, hr, min_max):
        super(EvalDataset, self).__init__()
        self.lr = lr
        self.hr = hr
        self.min_max = min_max

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx, :, :, :]).float()
        hr = torch.from_numpy(self.hr[idx, :, :, :]).float()
        min_max = self.min_max
        return lr, hr, min_max

    def __len__(self):
        return self.lr.shape[0]
