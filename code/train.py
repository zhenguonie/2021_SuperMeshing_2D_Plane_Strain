import argparse
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import RDN
from datasets import TrainDataset, EvalDataset, data_load
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, denormalize

class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)

class MaxMinDiffLoss(nn.Module):
    def __init__(self):
        super(MaxMinDiffLoss, self).__init__()
    
    def get_idx(self, img, type):
        if type == 'max':
            img_m = img.max()
            img_m_idx = torch.eq(img, img.max())
        elif type == 'min':
            img_m = img.min()
            img_m_idx = torch.eq(img, img.min())
#         print(img_m, img_m_idx)
        return img_m, img_m_idx

    def forward(self, sr, hr):
        sr_max = sr.max()
        hr_max = hr.max()
        return torch.pow(sr_max - hr_max, 2)
#         另一种最大值误差
#         sr_max, sr_max_idx = self.get_idx(sr, 'max')
#         sr_min, sr_min_idx = self.get_idx(sr, 'min')
#         hr_max, hr_max_idx = self.get_idx(hr, 'max')
#         hr_min, hr_min_idx = self.get_idx(hr, 'min')
# #         print(sr_max, hr[sr_max_idx], sr[sr_max_idx])
# #         print(hr_max, sr[hr_max_idx], hr[hr_max_idx])
#         self.loss_max = (torch.abs(sr_max - hr[sr_max_idx][0]) + torch.abs(hr_max - sr[hr_max_idx][0])) / 2
#         self.loss_min = (torch.abs(sr_min - hr[sr_min_idx][0]) + torch.abs(hr_min - sr[hr_min_idx][0])) / 2
#         return (self.loss_max + self.loss_min) / 2

def get_loss_dict():
    loss = {}
    if (abs(rec_w - 0) > 1e-8):
        loss['rec_loss'] = ReconstructionLoss(type='l1')
    if (abs(mmd_w - 0) > 1e-8):
        loss['mmd_loss'] = MaxMinDiffLoss()
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default="D:\\SuperMeshing\\data_10.18\\stress_16.csv")
    parser.add_argument('--label-file', type=str, default="D:\\SuperMeshing\\data_10.18\\stress_64.csv")
    parser.add_argument('--model-save-path', type=str, default="./ilk/x4_8_8")
    parser.add_argument('--result-path', type=str, default="./result/x4_8_8")
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-epochs', type=int, default=800)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--low-res', type=int, default=16)
    parser.add_argument('--high-res', type=int, default=64)
    args = parser.parse_args()

    # args.model_save_path = os.path.join(args.model_save_path, 'x{}'.format(args.scale))

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    
    # args.result_path = os.path.join(args.result_path, 'x{}'.format(args.scale))

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    rec_w = 1
    mmd_w = 0.01

    model = RDN(scale_factor=args.scale,
                num_channels=1,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    # criterion = nn.L1Loss()
    loss_all = get_loss_dict()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    Inputs_train, Targets_train, Inputs_valid, Targets_valid, Min_Max = \
        data_load(args.input_file, args.label_file, args.low_res, args.high_res, args.seed)

    train_dataset = TrainDataset(Inputs_train, Targets_train, scale=args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(Inputs_valid, Targets_valid, Min_Max)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    LOSSES = []
    REC_LOSSES = []
    MMD_LOSSES = []

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['learning_rate'] = args.learning_rate * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()
        epoch_rec_losses = AverageMeter()
        epoch_mmd_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                # loss = criterion(preds, labels)
                rec_loss = rec_w * loss_all['rec_loss'](preds[:,:,4:-4,4:-4], labels)
                mmd_loss = mmd_w * loss_all['mmd_loss'](preds[:,:,4:-4,4:-4], labels)
                loss = rec_loss + mmd_loss

                epoch_losses.update(loss.item(), len(inputs))
                epoch_rec_losses.update(rec_loss.item(), len(inputs))
                epoch_mmd_losses.update(mmd_loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.10f}, {:.10f}, {:.10f}'.format(epoch_losses.avg, epoch_rec_losses.avg, epoch_mmd_losses.avg))
                # t.set_postfix(loss='{:.10f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        
        LOSSES.append(epoch_losses.avg)
        REC_LOSSES.append(epoch_rec_losses.avg)
        MMD_LOSSES.append(epoch_mmd_losses.avg)
        
        with open(args.result_path + '/loss.txt', 'a') as f:
            f.write(str(epoch_losses.avg))
            f.write('\n')
        with open(args.result_path + '/rec_loss.txt', 'a') as f:
            f.write(str(epoch_rec_losses.avg))
            f.write('\n')
        with open(args.result_path + '/mmd_loss.txt', 'a') as f:
            f.write(str(epoch_mmd_losses.avg))
            f.write('\n')

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'epoch_{}.pth'.format(epoch + 1)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels, min_max = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs.shape, preds.shape, labels.shape)

            with torch.no_grad():
                preds = model(inputs)[:, :, args.scale:-args.scale, args.scale:-args.scale]

            mask = labels == 0.0
            preds.masked_fill(mask, 0.0)
            # print(inputs.shape, preds.shape, labels.shape)

            preds = denormalize(torch.squeeze(preds), min_max)
            labels = denormalize(torch.squeeze(labels), min_max)
            # print(inputs.shape, preds.shape, labels.shape)

            preds = preds[args.scale:-args.scale, args.scale:-args.scale]
            labels = labels[args.scale:-args.scale, args.scale:-args.scale]
            # print(inputs.shape, preds.shape, labels.shape)

            min_ = int(min_max[2].type(torch.cuda.FloatTensor).cpu().numpy())
            max_ = int(min_max[3].type(torch.cuda.FloatTensor).cpu().numpy())
            epoch_psnr.update(calc_psnr(preds, labels, min_=min_, max_=max_), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.model_save_path, 'best.pth'))

    LOSSES = np.array(LOSSES)
    REC_LOSSES = np.array(REC_LOSSES)
    MMD_LOSSES = np.array(MMD_LOSSES)
    plt.plot(LOSSES)
    plt.savefig(args.result_path + '/loss.png')
    plt.show()
    plt.plot(REC_LOSSES)
    plt.savefig(args.result_path + '/rec_loss.png')
    plt.show()
    plt.plot(MMD_LOSSES)
    plt.savefig(args.result_path + '/mmd_loss.png')
    plt.show()
