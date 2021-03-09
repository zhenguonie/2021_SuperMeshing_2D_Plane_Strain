import torch
import numpy as np
import os


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img, min_max):
    min_ = min_max[2].type(torch.cuda.FloatTensor)
    max_ = min_max[3].type(torch.cuda.FloatTensor)
    # print(img.dtype, min_.dtype, max_.dtype)
    img = img.mul(max_).clamp(int(min_.cpu().numpy()), int(max_.cpu().numpy()))
    return img


def denormalize_test(img, min_max):
    min_ = min_max[2]
    max_ = min_max[3]
    # print(img.dtype, min_.dtype, max_.dtype)
    img = img.mul(max_).clamp(int(min_), int(max_))
    return img


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def calc_psnr(img1, img2, max_=255.0, min_=0.0):
    return 10. * (((max_ - min_) ** 2) / ((img1 - img2) ** 2).mean()).log10()


def calc_mre(img1, img2, delta=0.01):
    diff = (img1 - img2).abs()
    return (diff / (delta + img1)).mean()


def calc_max_diff(img1, img2):
    img1_max = np.max(img1.cpu().numpy())
    img2_max = np.max(img2.cpu().numpy())
    return abs(img1_max - img2_max) / img1_max


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
