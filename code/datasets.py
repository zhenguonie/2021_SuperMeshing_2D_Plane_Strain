import random
import h5py
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


# 需要修改
path_1x = '/home/breeze/xuhanding/data/stress_32.csv'
path_8x = '/home/breeze/xuhanding/data/stress_256.csv'
train_ratio = 0.8


def data_load(path_1x, path_8x, train_ratio):
    Inputs = []
    Targets = []
    MM = []
    with open(path_1x, 'r') as csv_file_1x:
        with open(path_8x, 'r') as csv_file_8x:
            reader1 = csv.reader(csv_file_1x)
            reader2 = csv.reader(csv_file_8x)
            num = 0
            for row1, row2 in zip(reader1, reader2):
                if num <= 3240: # 数据总数,需调整
                    arr_1x = np.array([float(i) for i in row1])
                    arr_8x = np.array([float(i) for i in row2])

                    # mm = []
                    # mm.append(np.min(arr_1x))
                    # mm.append(np.max(arr_1x))
                    # mm.append(np.min(arr_8x))
                    # mm.append(np.max(arr_8x))
                    
                    # target_ = np.clip(arr_8x, mm[0], mm[1])
                    # input_ = np.array([(pixel - mm[0]) / (mm[1] - mm[0]) for pixel in arr_1x])
                    # target_ = np.array([(pixel - mm[0]) / (mm[1] - mm[0]) for pixel in arr_8x])
                    input_ = arr_1x.reshape([32, 32])[np.newaxis, :, :]
                    target_ = arr_8x.reshape([256, 256])[np.newaxis, :, :]

                    Inputs.append(input_)
                    Targets.append(target_)
                    # MM.append(mm)
                num += 1
                if num % 100 == 0:
                    print(num)
    
    lr_min = np.min(Inputs)
    lr_max = np.max(Inputs)
    min_ = lr_min * 3
    max_ = lr_max * 3
    MM = [lr_min, lr_max, min_, max_]
    print(MM)
    Inputs = (Inputs - min_) / (max_ - min_)
    Targets = (Targets - min_) / (max_ - min_)

    train_num = 500 # 需要修改
    print(train_num)

    state = np.random.get_state()
    np.random.shuffle(Inputs)
    np.random.set_state(state)
    np.random.shuffle(Targets)
    # np.random.set_state(state)
    # np.random.shuffle(MM)

    Inputs_train = np.array(Inputs[:train_num])
    Targets_train = np.array(Targets[:train_num])
    # MM_train = np.array(MM[:train_num])
    Inputs_valid = np.array(Inputs[3000:])
    Targets_valid = np.array(Targets[3000:])
    # MM_valid = np.array(MM[train_num:])
    print(Inputs_train.shape, Targets_train.shape)
    print(Inputs_valid.shape, Targets_valid.shape)
    print(Inputs_train[0, 0, :])
    
    return Inputs_train, Targets_train, Inputs_valid, Targets_valid, MM


class TrainDataset(Dataset):
    # def __init__(self, h5_file, patch_size, scale):
    def __init__(self, lr, hr, scale):
        super(TrainDataset, self).__init__()
        # self.h5_file = h5_file
        self.lr = lr
        self.hr = hr
        self.scale = scale

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    @staticmethod
    def random_rotate(lr, hr):
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
        # with h5py.File(self.h5_file, 'r') as f:
        #     lr = f['lr'][str(idx)][::]
        #     hr = f['hr'][str(idx)][::]
        #     lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
        #     lr, hr = self.random_horizontal_flip(lr, hr)
        #     lr, hr = self.random_vertical_flip(lr, hr)
        #     lr, hr = self.random_rotate_90(lr, hr)
        #     lr = lr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        #     hr = hr.astype(np.float32).transpose([2, 0, 1]) / 255.0
        #     return lr, hr
        lr, hr = self.random_rotate(self.lr[idx, :, :, :], self.hr[idx, :, :, :])
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        # lr = torch.from_numpy(self.lr[idx, :, :, :]).float()
        # hr = torch.from_numpy(self.hr[idx, :, :, :]).float()
        return lr, hr

    def __len__(self):
        # with h5py.File(self.h5_file, 'r') as f:
        #     return len(f['lr'])
        return self.lr.shape[0]


class EvalDataset(Dataset):
    def __init__(self, lr, hr, min_max):
    # def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        # self.h5_file = h5_file
        self.lr = lr
        self.hr = hr
        self.min_max = min_max

    def __getitem__(self, idx):
        # with h5py.File(self.h5_file, 'r') as f:
        #     lr = f['lr'][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) / 255.0
        #     hr = f['hr'][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) / 255.0
        #     return lr, hr
        lr = torch.from_numpy(self.lr[idx, :, :, :]).float()
        hr = torch.from_numpy(self.hr[idx, :, :, :]).float()
        min_max = self.min_max
        return lr, hr, min_max

    def __len__(self):
        # with h5py.File(self.h5_file, 'r') as f:
        #     return len(f['lr'])
        return self.lr.shape[0]
