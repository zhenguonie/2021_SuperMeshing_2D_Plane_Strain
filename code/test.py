import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.nn import L1Loss, MSELoss
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import time
import os

from models import RDN
from datasets import data_load
# from train import Min_Max
from utils import convert_rgb_to_y, denormalize_test, calc_psnr, calc_mre, calc_mre2, calc_max_diff, remove_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default="D:\\SuperMeshing\\data_10.18\\stress_16.csv")
    parser.add_argument('--label-file', type=str, default="D:\\SuperMeshing\\data_10.18\\stress_32.csv")
    parser.add_argument('--weights-file', type=str, default='./ilk/x2/epoch_800.pth')
    parser.add_argument('--result-path', type=str, default="./result/x2")
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--low-res', type=int, default=16)
    parser.add_argument('--high-res', type=int, default=32)
    args = parser.parse_args()
    
    # low_res=8
    # high_res=64

    # args.result_path = os.path.join(args.result_path, 'x{}'.format(args.scale))

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
        data_load(args.input_file, args.label_file, args.low_res, args.scale*args.low_res, args.seed)
    
    remove_file(args.result_path + '/time.txt')
    remove_file(args.result_path + '/psnr.txt')
    remove_file(args.result_path + '/mre.txt')
    remove_file(args.result_path + '/mre2.txt')
    remove_file(args.result_path + '/max_diff.txt')
    
    i = 0
    PSNR = []
    MRE = []
    MRE2 = []
    MRE2_lin = []
    MRE2_cub = []
    TIME = []
    MAX_DIFF = []
    MAX_DIFF_LIN = []
    MAX_DIFF_CUB = []

    MAE = []
    MAE_lin = []
    MAE_cub = []
    MSE = []
    MSE_lin = []
    MSE_cub = []
    for lr, hr in zip(Inputs_valid, Targets_valid):
        # if i >= 300:
        #     break
        
        start = time.time()

        lr = np.expand_dims(lr.astype(np.float32), 0)
        hr = np.expand_dims(hr.astype(np.float32), 0)
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)

        with torch.no_grad():
            preds = torch.squeeze(model(lr))[args.scale:-args.scale, args.scale:-args.scale]

        mask = hr == 0.0
        preds.masked_fill(mask, 0.0)
        
        # calculate process time
        end = time.time()
        time_pass = end - start
        print('Time: {:f}'.format(time_pass))
        TIME.append(time_pass)
        with open(args.result_path + '/time.txt', 'a') as f:
            f.write(str(time_pass))
            f.write('\n')
        
        b_lin = torch.nn.functional.interpolate(
            lr.view(1, 1, args.low_res, args.low_res), size=[args.high_res,args.high_res],  mode='bilinear').view(args.high_res,args.high_res)
        c_lin = torch.nn.functional.interpolate(
            lr.view(1, 1, args.low_res, args.low_res), size=[args.high_res,args.high_res],  mode='bicubic').view(args.high_res,args.high_res)
        
        # mae
        mae_fn = L1Loss()
        mae = mae_fn(preds, hr)
        mae_lin = mae_fn(b_lin, hr)
        mae_cub = mae_fn(c_lin, hr)
        print('MAE: ', mae.cpu().numpy())
        MAE.append(mae.cpu().numpy())
        with open(args.result_path + '/mae.txt', 'a') as f:
            f.write(str(mae.cpu().numpy()))
            f.write('\n')
        print('MAE_lin: ', mae_lin.cpu().numpy())
        MAE_lin.append(mae_lin.cpu().numpy())
        with open(args.result_path + '/mae_lin.txt', 'a') as f:
            f.write(str(mae_lin.cpu().numpy()))
            f.write('\n')
        print('MAE_cub: ', mae_cub.cpu().numpy())
        MAE_cub.append(mae_cub.cpu().numpy())
        with open(args.result_path + '/mae_cub.txt', 'a') as f:
            f.write(str(mae_cub.cpu().numpy()))
            f.write('\n')

        # mse
        mse_fn = MSELoss()
        mse = mse_fn(preds, hr)
        mse_lin = mse_fn(b_lin, hr)
        mse_cub = mse_fn(c_lin, hr)
        print('MSE: ', mse.cpu().numpy())
        MSE.append(mse.cpu().numpy())
        with open(args.result_path + '/mse.txt', 'a') as f:
            f.write(str(mse.cpu().numpy()))
            f.write('\n')
        print('MSE_lin: ', mse_lin.cpu().numpy())
        MSE_lin.append(mse_lin.cpu().numpy())
        with open(args.result_path + '/mse_lin.txt', 'a') as f:
            f.write(str(mse_lin.cpu().numpy()))
            f.write('\n')
        print('MSE_cub: ', mse_cub.cpu().numpy())
        MSE_cub.append(mse_cub.cpu().numpy())
        with open(args.result_path + '/mse_cub.txt', 'a') as f:
            f.write(str(mse_cub.cpu().numpy()))
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
        mre = calc_mre(hr, preds)
        print('MRE: ', mre.cpu().numpy())
        MRE.append(mre.cpu().numpy())
        with open(args.result_path + '/mre.txt', 'a') as f:
            f.write(str(mre.cpu().numpy()))
            f.write('\n')
        
        # calculate mean relative error 2
        mre2 = calc_mre2(hr, preds)
        print('MRE: ', mre2.cpu().numpy())
        MRE2.append(mre2.cpu().numpy())
        with open(args.result_path + '/mre2.txt', 'a') as f:
            f.write(str(mre2.cpu().numpy()))
            f.write('\n')

        # calculate maximum stress difference
        max_diff = calc_max_diff(hr, preds)
        print('max diff: {:f}'.format(max_diff))
        MAX_DIFF.append(max_diff)
        with open(args.result_path + '/max_diff.txt', 'a') as f:
            f.write(str(max_diff))
            f.write('\n')

        # output result
        lr = torch.squeeze(lr)
        lr = lr.mul(min_max[1] - min_max[0]).add(min_max[0]).clamp(min_max[0], min_max[1])

        # bilinear interpolation
        b_lin = torch.nn.functional.interpolate(
            lr.view(1, 1, args.low_res, args.low_res), size=[args.high_res,args.high_res],  mode='bilinear').view(args.high_res,args.high_res)
        mre2_lin = calc_mre2(hr, b_lin)
        print('MRE_lin: ', mre2_lin.cpu().numpy())
        MRE2_lin.append(mre2_lin.cpu().numpy())
        with open(args.result_path + '/mre2_lin.txt', 'a') as f:
            f.write(str(mre2_lin.cpu().numpy()))
            f.write('\n')
        max_diff_lin = calc_max_diff(hr, b_lin)
        print('max diff lin: {:f}'.format(max_diff_lin))
        MAX_DIFF_LIN.append(max_diff_lin)
        with open(args.result_path + '/max_diff_lin.txt', 'a') as f:
            f.write(str(max_diff_lin))
            f.write('\n')
        
        # cubic linear interpolation
        c_lin = torch.nn.functional.interpolate(
            lr.view(1, 1, args.low_res, args.low_res), size=[args.high_res,args.high_res],  mode='bicubic').view(args.high_res,args.high_res)
        mre2_cub = calc_mre2(hr, c_lin)
        print('MRE_cub: ', mre2_cub.cpu().numpy())
        MRE2_cub.append(mre2_cub.cpu().numpy())
        with open(args.result_path + '/mre2_cub.txt', 'a') as f:
            f.write(str(mre2_cub.cpu().numpy()))
            f.write('\n')
        max_diff_cub = calc_max_diff(hr, c_lin)
        print('max diff cub: {:f}'.format(max_diff_cub))
        MAX_DIFF_CUB.append(max_diff_cub)
        with open(args.result_path + '/max_diff_cub.txt', 'a') as f:
            f.write(str(max_diff_cub))
            f.write('\n')
        
        vmin = hr.cpu().min()
        vmax = hr.cpu().max()

        plt.rcParams['axes.facecolor'] = 'gray'
        fig, ax = plt.subplots(1, 3)
        plt.subplot(1,3,1)
        # plt.imshow(lr.cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        mask = lr.cpu() == 0.0
        masked_lr = np.ma.array(lr.cpu(), mask=mask)
        plt.imshow(masked_lr, cmap='jet')
        plt.xticks([0, 4, 8, 12, 15], fontsize=14)
        plt.yticks([0, 4, 8, 12, 15], fontsize=14)
        plt.subplot(1,3,2)
        # plt.imshow(hr.cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        mask = hr.cpu() == 0.0
        masked_hr = np.ma.array(hr.cpu(), mask=mask)
        plt.imshow(masked_hr, cmap='jet')
        plt.xticks([0, 8, 16, 24, 31], fontsize=14)
        plt.yticks([0, 8, 16, 24, 31], fontsize=14)
        plt.subplot(1,3,3)
        # plt.imshow(preds.cpu(), cmap='jet')
        mask = hr.cpu() == 0.0
        masked_preds = np.ma.array(preds.cpu(), mask=mask)
        plt.imshow(masked_preds, cmap='jet')
        plt.xticks([0, 8, 16, 24, 31], fontsize=14)
        plt.yticks([0, 8, 16, 24, 31], fontsize=14)
        # plt.subplot(1,4,4)
        # # plt.imshow(np.abs(preds.cpu() - hr.cpu()), cmap='jet')
        # mask = hr.cpu() == 0.0
        # masked_diff = np.ma.array(np.abs(preds.cpu() - hr.cpu()), mask=mask)
        # plt.imshow(masked_diff, cmap='jet', vmin=vmin, vmax=vmax)
        # plt.subplot(1,5,5)
        # plt.imshow(b_lin.cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.savefig(args.result_path + '/{}.png'.format(str(i+1)))
        i += 1
    
    time_sum = 0
    for i in range(10):
        time_sum += TIME[i]
    print('Time: {:f}s'.format(time_sum))

    print('MRE: ', np.array(MRE).mean())

    # # plot maximum difference disturbtion
    # statistic = {'0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0, '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 0, '0.9': 0, '1.0': 0}
    # for i in MAX_DIFF:
    #     if i < 0.001:
    #         statistic['0.1'] += 1
    #     elif i < 0.002:
    #         statistic['0.2'] += 1
    #     elif i < 0.003:
    #         statistic['0.3'] += 1
    #     elif i < 0.004:
    #         statistic['0.4'] += 1
    #     elif i < 0.005:
    #         statistic['0.5'] += 1
    #     elif i < 0.006:
    #         statistic['0.6'] += 1
    #     elif i < 0.007:
    #         statistic['0.7'] += 1
    #     elif i < 0.008:
    #         statistic['0.8'] += 1
    #     elif i < 0.009:
    #         statistic['0.9'] += 1
    #     else:
    #         statistic['1.0'] += 1
    # s_name = []
    # s_value = []
    # for i in statistic:
    #     s_value.append(statistic[i])
    #     s_name.append(i)
    # plt.figure(figsize=(8, 6))
    # plt.bar(s_name, s_value)
    # plt.xlabel('ARSE(%)', fontsize=24)
    # plt.ylabel('Sample numbers', fontsize=24)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.savefig(args.result_path + '/max_diff.png')
    # # plt.show()
