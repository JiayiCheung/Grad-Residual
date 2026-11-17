"""
One-dimensional regression target with an easy and a hard half.

    easy(x)    = sin(0.5 x) + 0.5 sin(2.5 x)
    hard(x; f) = sin(2 f x) + 0.5 sin(7 f x)
    orientation "right": easy on x < 0, hard on x >= 0;  "left": the mirrored target y(-x)

f = 1 with orientation "right" is the original target. Training and validation inputs are drawn uniformly on [-4 pi, 4 pi]
with Gaussian label noise; the test set is a noise-free grid. One independent data set per seed (leading axis).
"""
import math

import torch

N_TRAIN, N_VAL, N_TEST = 3000, 1000, 3000
NOISE_SD = 0.1
X_RANGE = 4 * math.pi


def target(x, f=1.0, orient="right"):
    def right(z):
        easy = torch.sin(0.5 * z) + 0.5 * torch.sin(2.5 * z)
        hard = torch.sin(2.0 * f * z) + 0.5 * torch.sin(7.0 * f * z)
        return torch.where(z < 0, easy, hard)

    return right(x) if orient == "right" else right(-x)


def make_data(gen, S, device, f=1.0, orient="right", noise_mult=1.0):
    """returns (X, Y, Xv, Yv, Xt, Yt), each of shape (S, n, 1)."""
    sd = NOISE_SD * float(noise_mult)
    # the order of the draws is part of the protocol: it keeps inputs and noise paired across targets
    X = (torch.rand(S, N_TRAIN, 1, generator=gen, device=device) * 2 - 1) * X_RANGE
    Y = target(X, f, orient) + sd * torch.randn(S, N_TRAIN, 1, generator=gen, device=device)
    Xv = (torch.rand(S, N_VAL, 1, generator=gen, device=device) * 2 - 1) * X_RANGE
    Yv = target(Xv, f, orient) + sd * torch.randn(S, N_VAL, 1, generator=gen, device=device)
    Xt = torch.linspace(-X_RANGE, X_RANGE, N_TEST, device=device).view(1, -1, 1).expand(S, -1, -1)
    Yt = target(Xt, f, orient)
    return X, Y, Xv, Yv, Xt, Yt


def hard_mask(Xt, orient):
    """boolean mask of the test points that lie on the hard half."""
    neg = Xt[0, :, 0] < 0
    return neg if orient == "left" else ~neg
