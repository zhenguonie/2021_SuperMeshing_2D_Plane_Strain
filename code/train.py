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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default="/home/breeze/xuhanding/data/stress_32.csv")
    parser.add_argument('--label-file', type=str, default="/home/breeze/xuhanding/data/stress_128.csv")
    parser.add_argument('--model-save-path', type=str, default="./ilk/2d_mises")
    parser.add_argument('--result-path', type=str, default="./result/2d_mises")
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=800)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.model_save_path = os.path.join(args.model_save_path, 'x{}'.format(args.scale))

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    
    args.result_path = os.path.join(args.result_path, 'x{}'.format(args.scale))

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = RDN(scale_factor=args.scale,
                num_channels=1,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), learning_rate=args.learning_rate)

    Inputs_train, Targets_train, Inputs_valid, Targets_valid, Min_Max = \
        data_load(args.input_path, args.label_path)

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

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['learning_rate'] = args.learning_rate * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        
        LOSSES.append(epoch_losses.avg)
        
        with open(args.result_path + '/loss.txt', 'a') as f:
            f.write(str(epoch_losses.avg))
            f.write('\n')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels, min_max = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            preds = denormalize(torch.squeeze(preds), min_max)
            labels = denormalize(torch.squeeze(labels), min_max)

            preds = preds[args.scale:-args.scale, args.scale:-args.scale]
            labels = labels[args.scale:-args.scale, args.scale:-args.scale]

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
    plt.plot(LOSSES)
    plt.savefig(args.result_path + '/loss.png')
    plt.show()
