"""Metric wrappers used by hessian_weights.py and vvc_rdo_experiment.py.

Trimmed copy of the authors' internal Common/utils/q_utils.py: the same objects and functions, with the same
window sizes and pooling, minus an unrelated cvxpy routine.  Kept under the original import path
(Common.utils.q_utils) so the method files are unchanged.
"""
import numpy as np
import torchvision
from DISTS_pytorch import DISTS
from pytorch_msssim import SSIM
from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d
import lpips
import torch

win_sigma = 1.5
truncate = 3.5
r = int(truncate * win_sigma + 0.5)  # radius as in ndimage
win_size = 2 * r + 1

ssim_module = SSIM(data_range=255, win_size=win_size, size_average=True, channel=1, win_sigma=win_sigma)  # channel=1 for grayscale images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_obj = lpips.LPIPS(net='vgg').to(device)

# create metric with default setting
D = DISTS()


def ycbcr_to_rgb(ycbcr_image):
    # Conversion matrix from YCbCr to RGB, transposed for correct matrix multiplication
    mat = torch.tensor(np.array([[1.0, 1.0, 1.0],
                        [0.0, -0.344136, 1.772],
                        [1.402, -0.714136, 0.0]]).T).to(ycbcr_image.device).float()
    # Bias to be subtracted before conversion
    bias = torch.tensor([0.0, 128.0, 128.0]).to(ycbcr_image.device).float()
    bias = bias.reshape((1, 3, 1, 1))  # Reshape for broadcasting
    ycbcr_permuted = (ycbcr_image - bias).permute(0, 2, 3, 1)  # [N, H, W, C]
    rgb_permuted = ycbcr_permuted @ mat.T
    rgb_image = rgb_permuted.permute(0, 3, 1, 2)
    return rgb_image


def compute_LPIPS_yuv(img1, img2):
    rgb1 = ycbcr_to_rgb(img1)
    rgb2 = ycbcr_to_rgb(img2)
    img1 = 2 * (rgb1 / 255 - 0.5)
    img2 = 2 * (rgb2 / 255 - 0.5)
    dist = lpips_obj(img1, img2)
    return dist


def compute_LPIPS_rgb(rgb1, rgb2):
    img1 = 2 * (rgb1 / 255 - 0.5)
    img2 = 2 * (rgb2 / 255 - 0.5)
    dist = lpips_obj(img1, img2)
    return dist


def dists_func(img1, img2):
    rgb1 = ycbcr_to_rgb(img1)
    rgb2 = ycbcr_to_rgb(img2)
    img1 = rgb1 / 255
    img2 = rgb2 / 255
    return D(img1, img2, require_grad=True, batch_average=True)


def preprocess(img, win_size=9):
    pad_int = (win_size-1)
    img = torchvision.transforms.Pad(padding=pad_int, padding_mode='symmetric')(img)
    return img


def ssim_func(img1, img2):
    pool_filt = torch.nn.functional.avg_pool2d
    ker_size = np.max([img1.shape[2], img1.shape[3]]) // 256
    if ker_size > 0:
        img1 = preprocess(img1, 2)
        img2 = preprocess(img2, 2)
        padding = 0
        img1 = pool_filt(img1, kernel_size=ker_size, padding=padding)
        img2 = pool_filt(img2, kernel_size=ker_size, padding=padding)
    pad = (11-1)//2+1
    img1 = preprocess(img1, pad)
    img2 = preprocess(img2, pad)
    return 1-ssim_module(img1, img2)


def ms_ssim_func(img1, img2):
    pool_filt = torch.nn.functional.avg_pool2d
    levels = 5
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
    msssim = []
    win_sigma = 3
    win_size = 11
    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([img1.shape[1]] + [1] * (len(img1.shape) - 1))
    pad = (win_size-1)//2+1
    for i in range(levels):
        img1_1 = preprocess(img1, pad)
        img2_1 = preprocess(img2, pad)
        ssim_per_channel, cs = _ssim(img1_1, img2_1, data_range=255, size_average=False, win=win, K=(0.01, 0.03))
        if i < levels - 1:
            msssim.append(cs)
            padding = [s % 2 for s in img1.shape[2:]]
            img1 = pool_filt(img1, kernel_size=2, padding=padding)
            img2 = pool_filt(img2, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(msssim + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    return 1-ms_ssim_val.mean()
