import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import time
import os

from models import RDN
from datasets import data_load
from utils import convert_rgb_to_y, denormalize_test, calc_psnr, calc_mre, calc_max_diff, remove_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default="/home/breeze/xuhanding/data/stress_32.csv")
    parser.add_argument('--label-file', type=str, default="/home/breeze/xuhanding/data/stress_128.csv")
    parser.add_argument('--weights-file', type=str, default='./ilk/2d_mises/x4/epoch_799.pth')
    parser.add_argument('--result-path', type=str, default="./result/2d_mises")
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    
    args.result_path = os.path.join(args.result_path, 'x{}'.format(args.scale))

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RDN(scale_factor=args.scale,
                num_channels=1,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    Inputs_train, Targets_train, Inputs_valid, Targets_valid, min_max = \
        data_load(args.input_path, args.label_path)
    
    remove_file(args.result_path + '/time.txt')
    remove_file(args.result_path + '/psnr.txt')
    remove_file(args.result_path + '/mre.txt')
    remove_file(args.result_path + '/max_diff.txt')
    
    i = 0
    PSNR = []
    MRE = []
    TIME = []
    MAX_DIFF = []
    for lr, hr in zip(Inputs_valid, Targets_valid):
        
        start = time.time()

        lr = np.expand_dims(lr.astype(np.float32), 0)
        hr = np.expand_dims(hr.astype(np.float32), 0)
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)

        with torch.no_grad():
            preds = torch.squeeze(model(lr))
        
        # calculate process time
        end = time.time()
        time_pass = end - start
        print('Time: {:f}'.format(time_pass))
        TIME.append(time_pass)
        with open(args.result_path + '/time.txt', 'a') as f:
            f.write(str(time_pass))
            f.write('\n')

        preds = denormalize_test(torch.squeeze(preds), min_max)
        hr = denormalize_test(torch.squeeze(hr), min_max)

        pred = preds[args.scale:-args.scale, args.scale:-args.scale]
        label = hr[args.scale:-args.scale, args.scale:-args.scale]

        # calculate PSNR
        psnr = -calc_psnr(label, pred)
        print('PSNR: {:.2f}'.format(psnr))
        PSNR.append(psnr.cpu().numpy())
        with open(args.result_path + '/psnr.txt', 'a') as f:
            f.write(str(psnr.cpu().numpy()))
            f.write('\n')

        # calculate mean relative error
        mre = calc_mre(label, pred)
        print('MRE: ', mre.cpu().numpy())
        MRE.append(mre.cpu().numpy())
        with open(args.result_path + '/mre.txt', 'a') as f:
            f.write(str(mre.cpu().numpy()))
            f.write('\n')

        # calculate maximum stress difference
        max_diff = calc_max_diff(label, pred)
        print('max diff: {:f}'.format(max_diff))
        MAX_DIFF.append(max_diff)
        with open(args.result_path + '/max_diff.txt', 'a') as f:
            f.write(str(max_diff))
            f.write('\n')

        # output result
        lr = torch.squeeze(lr)
        lr = lr.mul(min_max[1]).clamp(min_max[0], min_max[1])

        fig, ax = plt.subplots(1, 4)
        plt.subplot(1,4,1)
        plt.imshow(lr.cpu(), cmap='jet')
        plt.subplot(1,4,2)
        plt.imshow(hr.cpu(), cmap='jet')
        plt.subplot(1,4,3)
        plt.imshow(preds.cpu(), cmap='jet')
        plt.subplot(1,4,4)
        plt.imshow(np.abs(preds.cpu() - hr.cpu()), cmap='jet')
        plt.tight_layout()
        plt.savefig(args.result_path + '/{}.png'.format(str(i+1)))
        i += 1
    
    time_sum = 0
    for i in range(10):
        time_sum += TIME[i]
    print('Time: {:f}s'.format(time_sum))

    print('MRE: ', np.array(MRE).mean())

    # plot maximum difference disturbtion
    statistic = {'0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0, '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 0, '0.9': 0, '1.0': 0}
    for i in MAX_DIFF:
        if i < 0.001:
            statistic['0.1'] += 1
        elif i < 0.002:
            statistic['0.2'] += 1
        elif i < 0.003:
            statistic['0.3'] += 1
        elif i < 0.004:
            statistic['0.4'] += 1
        elif i < 0.005:
            statistic['0.5'] += 1
        elif i < 0.006:
            statistic['0.6'] += 1
        elif i < 0.007:
            statistic['0.7'] += 1
        elif i < 0.008:
            statistic['0.8'] += 1
        elif i < 0.009:
            statistic['0.9'] += 1
        else:
            statistic['1.0'] += 1
    s_name = []
    s_value = []
    for i in statistic:
        s_value.append(statistic[i])
        s_name.append(i)
    plt.figure(figsize=(8, 6))
    plt.bar(s_name, s_value)
    plt.xlabel('ARSE(%)', fontsize=24)
    plt.ylabel('Sample numbers', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(args.result_path + '/max_diff.png')
    plt.show()
