"""
Three-hidden-layer MLP with the shortcut / gradient term added inside the last ReLU, trained as many independent networks at once.

    h1 = ReLU(W1 x + b1),  h2 = ReLU(W2 h1 + b2),  z = W3 h2 + b3
    h3 = ReLU(z + s * q + c * g_hat),   y = Wo h3 + bo

    x_b   = h1 (variant F) or h2 (variant D): the shortcut source and the point where the gradient is taken
    q     = x_b, or x_b / max(||x_b||, 1e-2) for the normalised shortcut
    g     = d sum_j ReLU(z_j) / d x_b,  g_hat = g / (||g|| + 1e-7), a constant for the optimiser

Methods (the coefficients are per network, so one process can hold several configurations of one method):
    regular          s = c = 0
    standard_fixed   s fixed                     standard_train   s trainable (start s0)
    grad_only        c trainable (start c0)      grad_fixed       c fixed (optionally with g_hat shuffled across samples)
    addition         s fixed, c trainable or fixed at 1
    convex           s = 1 - sigmoid(alpha), c = sigmoid(alpha), alpha trainable
    grad_mag         h3 = [ReLU(z); ||g||]  (the length of g as one extra feature)

All networks of a process share one draw of the initial weights per seed (repeated over the blocks), so every block is paired.
"""
import hashlib
import math

import torch

EPS = 1e-7
Q_FLOOR = 1e-2
VARIANTS = {"F": 2, "D": 1}  # variant -> span: the shortcut source is hidden layer 3 - span
METHODS = ("regular", "standard_fixed", "standard_train", "grad_only", "grad_fixed", "addition", "convex", "grad_mag")
GRAD_METHODS = ("grad_only", "grad_fixed", "addition", "convex", "grad_mag")


def digest(*ts):
    h = hashlib.sha1()
    for t in ts:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _layer(gen, S, fan_in, fan_out, device):
    # pytorch's default nn.Linear init
    bound = 1.0 / math.sqrt(fan_in)
    W = (torch.rand(S, fan_in, fan_out, generator=gen, device=device) * 2 - 1) * bound
    b = (torch.rand(S, 1, fan_out, generator=gen, device=device) * 2 - 1) * bound
    return W, b


def draw_weights(gen, S, width, grad_mag=False, device="cpu"):
    """initial weights of S networks. the draw order does not depend on the method, so all methods share W1 ... b3 exactly."""
    d = width
    W1, b1 = _layer(gen, S, 1, d, device)
    W2, b2 = _layer(gen, S, d, d, device)
    W3, b3 = _layer(gen, S, d, d, device)
    # one readout draw of d + 1 rows; grad_mag uses all of them, the others the first d
    R = torch.rand(S, d + 1, 1, generator=gen, device=device) * 2 - 1
    rb = torch.rand(S, 1, 1, generator=gen, device=device) * 2 - 1
    bound = 1.0 / math.sqrt(d + 1 if grad_mag else d)
    Wo = R * bound if grad_mag else R[:, :d] * bound
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, Wo=Wo, bo=rb * bound)


def shuffle_nonzero(gn, gen):
    """permute the rows of gn (S, N, F) among the non-zero rows of each seed; zero rows stay zero, so every norm is kept."""
    S, N, F = gn.shape
    dev = gn.device
    nz = gn.norm(dim=-1) > 0
    ar = torch.arange(N, device=dev).unsqueeze(0).expand(S, N)
    keys = torch.where(nz, torch.rand(S, N, generator=gen, device=dev), torch.full((S, N), 2.0, device=dev))
    src_order = keys.argsort(dim=1)
    dest_order = torch.where(nz, ar, ar + N).argsort(dim=1)
    count = nz.sum(dim=1, keepdim=True)
    valid = ar < count
    perm = ar.clone()
    perm.scatter_(1, dest_order, torch.where(valid, src_order, dest_order))
    return gn.gather(1, perm.unsqueeze(-1).expand(S, N, F))


class StackedNet:
    """configurations of ONE method x blocks x S seeds stacked on the leading axis (nb = n_configs * blocks_per_config)."""

    def __init__(self, configs, weights, blocks_per_config, S, span, device, shortcut_norm="none", shuffle_gen=None):
        self.method = configs[0]["method"]
        assert self.method in METHODS and all(c["method"] == self.method for c in configs)
        self.span, self.shortcut_norm, self.shuffle_gen = span, shortcut_norm, shuffle_gen
        nb = len(configs) * blocks_per_config
        self.nb, self.S = nb, S
        rep = lambda t: t.detach().repeat(nb, 1, 1).contiguous().requires_grad_(True)
        for name in ("W1", "b1", "W2", "b2", "W3", "b3", "Wo", "bo"):
            setattr(self, name, rep(weights[name]))
        per = lambda vals: torch.tensor([v for v in vals for _ in range(blocks_per_config)], device=device, dtype=torch.float32).repeat_interleave(S).view(-1, 1, 1)
        m = self.method
        self.alpha, self.mask = None, None
        if m == "standard_fixed":
            self.s_fix = per([c["init"] for c in configs])
        elif m == "addition":
            self.s_fix = per([c["init2"] if c["init2"] is not None else 1.0 for c in configs])
        elif m == "grad_fixed":
            self.s_fix = per([0.0 for _ in configs])
        if m in ("standard_train", "grad_only", "convex"):
            self.alpha = per([c["init"] for c in configs]).clone().requires_grad_(True)
        elif m in ("addition", "grad_fixed"):
            # c per network: addition trains c where a start value is given (else c = 1 fixed); grad_fixed keeps c fixed
            cvals = [(c["init"] if c["init"] is not None else 1.0) if m == "addition" else c["c_fixed"] for c in configs]
            train = [(c["init"] is not None) if m == "addition" else False for c in configs]
            self.alpha = per(cvals).clone().requires_grad_(True)
            self.mask = per([1.0 if t else 0.0 for t in train])
            if not any(train):
                self.alpha.requires_grad_(False)
        self.trainable_alpha = self.alpha is not None and self.alpha.requires_grad

    def params(self):
        ps = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.Wo, self.bo]
        if self.trainable_alpha:
            ps.append(self.alpha)
        return ps

    def _source(self, xb):
        if self.shortcut_norm == "none":
            return xb
        # the gradient flows through the norm; the floor bounds the jacobian of near-dead samples
        n = xb.norm(dim=-1, keepdim=True).clamp_min(Q_FLOOR)
        return xb * (1.0 / n)

    def __call__(self, x):
        m = self.method
        h1 = torch.relu(x @ self.W1 + self.b1)
        h2 = torch.relu(h1 @ self.W2 + self.b2)
        z = h2 @ self.W3 + self.b3
        xb = h2 if self.span == 1 else h1
        h = torch.relu(z)
        if m in GRAD_METHODS:
            # g is taken without create_graph: g_hat is a constant for the optimiser
            g = torch.autograd.grad(h.sum(), xb, retain_graph=True)[0]
            norm = g.norm(dim=-1, keepdim=True)
            gn = g / (norm + EPS)
            if self.shuffle_gen is not None:
                gn = shuffle_nonzero(gn, self.shuffle_gen)
        if m == "regular":
            u = h
        elif m == "grad_mag":
            u = torch.relu(torch.cat([z, norm], dim=-1))
        else:
            if m == "standard_fixed":
                s = self.s_fix * self._source(xb)
            elif m == "standard_train":
                s = self.alpha * self._source(xb)
            elif m == "grad_only":
                s = self.alpha * gn
            elif m in ("addition", "grad_fixed"):
                s = self.s_fix * self._source(xb) + self.alpha * gn
            else:
                a = torch.sigmoid(self.alpha)
                s = (1 - a) * self._source(xb) + a * gn
            u = torch.relu(z + s)
        return u @ self.Wo + self.bo
