"""Functorch-safe MS-SSIM: identical arithmetic to pytorch_msssim.ms_ssim with the defaults msssim_std uses,
but with the window and the level weights passed in already on the right device and dtype.

The library builds both inside the call, via `win.to(X.device)` and `X.new_tensor(weights)`.  Under a
torch.func transform X is a wrapped tensor with no device, so those two lines raise
"DispatchKey FuncTorchGradWrapper doesn't correspond to a device" and forward-mode differentiation is
impossible.  Nothing else in the computation is unsafe, so this module hoists exactly those two lines out.
Verified against pytorch_msssim.ms_ssim in test_matches().
"""
import torch
import torch.nn.functional as F

WIN_SIZE, WIN_SIGMA = 11, 1.5
K1, K2 = 0.01, 0.03
LEVEL_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def make_win(channels, device, dtype):
    coords = torch.arange(WIN_SIZE, dtype=torch.float, device=device) - WIN_SIZE // 2
    g = torch.exp(-(coords ** 2) / (2 * WIN_SIGMA ** 2))
    g = (g / g.sum()).unsqueeze(0).unsqueeze(0)          # [1,1,W]
    # the library repeats a 3-D kernel against a 4-D input, which prepends a dim: [C,1,1,W].
    # _blur then transposes dim 2 with the last one to get the vertical pass.
    return g.repeat([channels, 1, 1, 1]).to(dtype)


def _blur(x, win):
    c = x.shape[1]
    out = x
    for i, s in enumerate(x.shape[2:]):
        if s >= win.shape[-1]:
            out = F.conv2d(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=c)
    return out


def _ssim(X, Y, win, data_range):
    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2
    mu1, mu2 = _blur(X, win), _blur(Y, win)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = _blur(X * X, win) - mu1_sq
    s2 = _blur(Y * Y, win) - mu2_sq
    s12 = _blur(X * Y, win) - mu1_mu2
    cs_map = (2 * s12 + c2) / (s1 + s2 + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return torch.flatten(ssim_map, 2).mean(-1), torch.flatten(cs_map, 2).mean(-1)


def ms_ssim_fs(X, Y, win, weights, data_range=255):
    """win: [C,1,1,WIN_SIZE] on X's device/dtype (make_win); weights: [5] likewise."""
    mcs = []
    for i in range(weights.shape[0]):
        ssim_per_channel, cs = _ssim(X, Y, win, data_range)
        if i < weights.shape[0] - 1:
            mcs.append(torch.relu(cs))
            pad = [s % 2 for s in X.shape[2:]]
            X = F.avg_pool2d(X, kernel_size=2, padding=pad)
            Y = F.avg_pool2d(Y, kernel_size=2, padding=pad)
    ssim_per_channel = torch.relu(ssim_per_channel)
    stack = torch.stack(mcs + [ssim_per_channel], dim=0)
    return torch.prod(stack ** weights.view(-1, 1, 1), dim=0).mean()


def test_matches(device='cuda'):
    from msssim_std import ms_ssim_std
    torch.manual_seed(0)
    for _ in range(3):
        a = torch.rand(1, 1, 512, 768, device=device) * 255
        b = (a + torch.randn_like(a) * 12).clamp(0, 255)
        win = make_win(1, a.device, a.dtype)
        w = torch.tensor(LEVEL_WEIGHTS, device=a.device, dtype=a.dtype)
        r1 = float(ms_ssim_std(a, b))
        r2 = float(ms_ssim_fs(a, b, win, w))
        print(f'  library {r1:.10f}   ours {r2:.10f}   diff {abs(r1 - r2):.2e}')


if __name__ == '__main__':
    test_matches()
