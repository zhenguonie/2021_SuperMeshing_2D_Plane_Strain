import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt
import time

from models import RDN
from datasets import data_load
from utils import convert_rgb_to_y, denormalize_test, calc_psnr, calc_mre, calc_max_diff, remove_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights-file', type=str, required=True)
    # parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--weights-file', type=str, default='./ilk/2d_mises/x8/epoch_799.pth')  # 需要修改
    parser.add_argument('--image-file', type=str, default='')
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=8)
    args = parser.parse_args()

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

    path_1x = '/home/breeze/xuhanding/data/stress_32.csv'
    path_8x = '/home/breeze/xuhanding/data/stress_256.csv'
    train_ratio = 0.8
    Inputs_train, Targets_train, Inputs_valid, Targets_valid, min_max = \
        data_load(path_1x, path_8x, train_ratio)
    
    remove_file('./result/2d_mises/x{}/time.txt'.format(args.scale))
    remove_file('./result/2d_mises/x{}/psnr.txt'.format(args.scale))
    remove_file('./result/2d_mises/x{}/mre.txt'.format(args.scale))
    remove_file('./result/2d_mises/x{}/max_diff.txt'.format(args.scale))
    
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

        # image = pil_image.open(args.image_file).convert('RGB')

        # image_width = (image.width // args.scale) * args.scale
        # image_height = (image.height // args.scale) * args.scale

        # hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        # lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        # bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        # bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

        # lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        # hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        # lr = torch.from_numpy(lr).to(device)
        # hr = torch.from_numpy(hr).to(device)

        with torch.no_grad():
            # preds = model(lr).squeeze(0)
            preds = torch.squeeze(model(lr))
        
        # 计算时间
        end = time.time()
        time_pass = end - start
        print('Time: {:f}'.format(time_pass))
        TIME.append(time_pass)
        with open('./result/2d_mises/x{}/time.txt'.format(args.scale), 'a') as f:
            f.write(str(time_pass))
            f.write('\n')

        # preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
        # hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

        # preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
        # hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

        # psnr = calc_psnr(hr_y, preds_y)
        preds = denormalize_test(torch.squeeze(preds), min_max)
        hr = denormalize_test(torch.squeeze(hr), min_max)

        pred = preds[args.scale:-args.scale, args.scale:-args.scale]
        label = hr[args.scale:-args.scale, args.scale:-args.scale]

        # 计算PSNR
        psnr = -calc_psnr(label, pred)
        print('PSNR: {:.2f}'.format(psnr))
        PSNR.append(psnr.cpu().numpy())
        with open('./result/2d_mises/x{}/psnr.txt'.format(args.scale), 'a') as f:
            f.write(str(psnr.cpu().numpy()))
            f.write('\n')

        # 计算MRE
        mre = calc_mre(label, pred)
        print('MRE: ', mre.cpu().numpy())
        MRE.append(mre.cpu().numpy())
        with open('./result/2d_mises/x{}/mre.txt'.format(args.scale), 'a') as f:
            f.write(str(mre.cpu().numpy()))
            f.write('\n')

        # 计算最大值误差
        max_diff = calc_max_diff(label, pred)
        print('max diff: {:f}'.format(max_diff))
        MAX_DIFF.append(max_diff)
        with open('./result/2d_mises/x{}/max_diff.txt'.format(args.scale), 'a') as f:
            f.write(str(max_diff))
            f.write('\n')

        # output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        # output.save(args.image_file.replace('.', '_rdn_x{}.'.format(args.scale)))

        # 输出结果图
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
        plt.savefig('./result/2d_mises/' + 'x{}/'.format(args.scale) + str(i+1) + '.png')
        i += 1
    
    time_sum = 0
    for i in range(10):
        time_sum += TIME[i]
    print('Time: {:f}s'.format(time_sum))

    print('MRE: ', np.array(MRE).mean())

    # 绘制最大值分布范围
    statistic = {'a': 0, 'b': 0, 'c': 0}
    for i in MAX_DIFF:
        if i < 0.1:
            statistic['a'] += 1
        elif i < 0.2:
            statistic['b'] += 1
        else:
            statistic['c'] += 1
    s_name = []
    s_value = []
    for i in statistic:
        s_value.append(statistic[i])
        s_name.append(i)
    plt.bar(s_name, s_value)
    plt.savefig('./result/2d_mises/x{}/max_diff.png'.format(args.scale))
