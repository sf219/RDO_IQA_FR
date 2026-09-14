"""Standard MS-SSIM for the ICASSP protocol (decided 4 Sept 2026): pytorch_msssim.ms_ssim with its defaults
(11-tap Gaussian window, sigma = 1.5, K = (0.01, 0.03), weights of Wang et al. 2003, valid convolution, avg-pool
downsampling), on the luma plane in [0, 255].  Replaces Common.utils.q_utils.ms_ssim_func (sigma = 3, symmetric
padding) in both the map generator and the evaluator.  Differentiable (autograd), so Hessian-vector products work."""
from pytorch_msssim import ms_ssim


def ms_ssim_std(img1, img2):
    return ms_ssim(img1, img2, data_range=255, size_average=True)
