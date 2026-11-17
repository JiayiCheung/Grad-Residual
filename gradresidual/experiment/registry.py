"""
The registered experiments. Each one is a function returning its list of units; the list is frozen into a manifest before any training.

lr_budget           every method at its own best: 7 methods (38 configurations per variant) x 6 learning rates x variants F, D x widths 8, 16, 32
                    x f = 1, 2 x both orientations; 30 seeds, 20000 epochs (5000 / 10000 / 20000 read from the same curves)
shortcut_scale      Standard Residual with the raw shortcut (s = 1), the shortcut scaled by a constant s* and the per-sample normalised shortcut
shuffled_direction  Grad Residual (c = 1) with the true g_hat and with g_hat shuffled across samples, against Regular
norm_shortcut       plan 1A: the lr_budget winners (drop rule, 20000 epochs) of the four shortcut methods, retrained with the normalised shortcut on
                    the same seeds, so every network is paired with its raw-shortcut twin in lr_budget
indep_grid          plan 1B: s and c both trainable on the normalised shortcut and g_hat (16 start pairs x 6 learning rates, widths 8 ... 64)

Units of the plan-1 experiments carry save_weights: all weights are written at these epochs (one .pt per unit).
"""
import math

from .units import make_unit

LRS = (0.5, 0.125, 0.03125, 0.0078125, 0.001953125, 0.00048828125)
WIDTHS = (8, 16, 32)

# s* = 1 / typical ||x_b|| of trained networks (calibrated once, before the runs that use it), per variant and width
S_STAR = {("F", 8): 0.13501, ("F", 16): 0.11327, ("F", 32): 0.08225, ("D", 8): 0.2959, ("D", 16): 0.24259, ("D", 32): 0.18554}


def coefficient_grid(width):
    return [0.01, 0.1, 1.0 / math.sqrt(width), 1.0]


def method_configs(width, variant):
    """(method, init, init2, c_fixed): init = s or s0 or c0 or alpha0, init2 = the fixed s of addition, c_fixed = the fixed c of grad_fixed."""
    s4 = coefficient_grid(width)
    out = []
    # regular has no shortcut, so it is the same network for F and D: trained once, under F
    if variant == "F":
        out.append(("regular", None, None, None))
    out += [("standard_fixed", s, None, None) for s in s4]
    out += [("standard_train", s, None, None) for s in s4]
    out += [("grad_only", c, None, None) for c in s4]
    out += [("grad_fixed", None, None, c) for c in (0.5, 1.0, 2.0)]
    out += [("addition", c, s, None) for c in s4 for s in s4]
    out.append(("addition", None, 1.0, None))
    out += [("convex", a, None, None) for a in (-3.0, -1.0, 1.0, 3.0)]
    out.append(("grad_mag", None, None, None))
    return out


def lr_budget(widths=WIDTHS, fs=(1.0, 2.0), seed0=5000, S=30, epochs=20000):
    out = []
    for w in widths:
        for v in ("F", "D"):
            for f in fs:
                for (m, i, i2, cf) in method_configs(w, v):
                    out.append(make_unit("lr_budget", v, w, f, m, i, i2, cf, lrs=LRS, S=S, seed0=seed0, epochs=epochs))
    return out


def shortcut_scale(widths=WIDTHS, lrs=(0.5, 0.125, 0.03125, 0.0078125), seed0=3000, S=100, epochs=5000):
    out = []
    for w in widths:
        for v in ("F", "D"):
            kw = dict(lrs=lrs, orients=("right",), S=S, seed0=seed0, epochs=epochs)
            out.append(make_unit("shortcut_scale", v, w, 1.0, "standard_fixed", 1.0, **kw))
            out.append(make_unit("shortcut_scale", v, w, 1.0, "standard_fixed", S_STAR[(v, w)], **kw))
            out.append(make_unit("shortcut_scale", v, w, 1.0, "standard_fixed", 1.0, shortcut_norm="unit", **kw))
    return out


def shuffled_direction(widths=WIDTHS, lr=0.0078125, seed0=2000, S=100, epochs=5000):
    out = []
    for w in widths:
        kw = dict(lrs=(lr,), S=S, seed0=seed0, epochs=epochs)
        out.append(make_unit("shuffled_direction", "F", w, 1.0, "regular", **kw))
        for v in ("F", "D"):
            out.append(make_unit("shuffled_direction", v, w, 1.0, "grad_fixed", c_fixed=1.0, **kw))
            out.append(make_unit("shuffled_direction", v, w, 1.0, "grad_fixed", c_fixed=1.0, shuffle=True, **kw))
    return out


SNAPSHOTS = [5000, 10000, 20000]

# lr_budget winners at 20000 epochs under the drop rule (crashed seeds dropped), shortcut families only:
# (variant, width, method, init, init2, lr, the f values it was selected at)
R8, R16, R32 = 1 / math.sqrt(8), 0.25, 1 / math.sqrt(32)
NORM_SHORTCUT_CONFIGS = [
    ("F", 8, "standard_fixed", R8, None, 0.03125, (1.0,)), ("F", 8, "standard_fixed", 0.01, None, 0.03125, (2.0,)),
    ("F", 8, "standard_train", R8, None, 0.03125, (1.0, 2.0)),
    ("F", 8, "addition", 0.1, R8, 0.03125, (1.0,)), ("F", 8, "addition", 0.1, 0.1, 0.03125, (2.0,)),
    ("F", 8, "convex", 1.0, None, 0.03125, (1.0, 2.0)),
    ("F", 16, "standard_fixed", 0.1, None, 0.03125, (1.0, 2.0)),
    ("F", 16, "standard_train", 0.1, None, 0.03125, (1.0,)), ("F", 16, "standard_train", R16, None, 0.03125, (2.0,)),
    ("F", 16, "addition", 1.0, 0.1, 0.03125, (1.0,)), ("F", 16, "addition", 0.01, R16, 0.03125, (2.0,)),
    ("F", 16, "convex", 3.0, None, 0.03125, (1.0, 2.0)),
    ("F", 32, "standard_fixed", 0.1, None, 0.03125, (1.0, 2.0)),
    ("F", 32, "standard_train", 0.01, None, 0.03125, (1.0,)), ("F", 32, "standard_train", R32, None, 0.03125, (2.0,)),
    ("F", 32, "addition", 1.0, 0.1, 0.03125, (1.0, 2.0)),
    ("F", 32, "convex", 3.0, None, 0.03125, (1.0, 2.0)),
    ("D", 8, "standard_fixed", R8, None, 0.03125, (1.0,)), ("D", 8, "standard_fixed", 1.0, None, 0.03125, (2.0,)),
    ("D", 8, "standard_train", R8, None, 0.03125, (1.0,)), ("D", 8, "standard_train", 0.01, None, 0.03125, (2.0,)),
    ("D", 8, "addition", None, 1.0, 0.03125, (1.0,)), ("D", 8, "addition", 1.0, R8, 0.03125, (2.0,)),
    ("D", 8, "convex", 3.0, None, 0.03125, (1.0,)), ("D", 8, "convex", 1.0, None, 0.03125, (2.0,)),
    ("D", 16, "standard_fixed", 0.1, None, 0.03125, (1.0, 2.0)),
    ("D", 16, "standard_train", R16, None, 0.03125, (1.0,)), ("D", 16, "standard_train", 0.1, None, 0.03125, (2.0,)),
    ("D", 16, "addition", 0.01, R16, 0.03125, (1.0,)), ("D", 16, "addition", 0.1, R16, 0.03125, (2.0,)),
    ("D", 16, "convex", 1.0, None, 0.03125, (1.0, 2.0)),
    ("D", 32, "standard_fixed", R32, None, 0.03125, (1.0, 2.0)),
    ("D", 32, "standard_train", 0.01, None, 0.03125, (1.0,)), ("D", 32, "standard_train", R32, None, 0.03125, (2.0,)),
    ("D", 32, "addition", 1.0, 0.1, 0.03125, (1.0,)), ("D", 32, "addition", R32, R32, 0.03125, (2.0,)),
    ("D", 32, "convex", 3.0, None, 0.03125, (1.0,)), ("D", 32, "convex", 1.0, None, 0.03125, (2.0,)),
]


def _snap(units):
    for u in units:
        u["save_weights"] = list(SNAPSHOTS)
    return units


def norm_shortcut(seed0=5000, S=30, epochs=20000):
    out = []
    for (v, w, m, i, i2, lr, fs) in NORM_SHORTCUT_CONFIGS:
        for f in fs:
            out.append(make_unit("norm_shortcut", v, w, f, m, i, i2, shortcut_norm="unit", lrs=(lr,), S=S, seed0=seed0, epochs=epochs))
    return _snap(out)


def indep_grid(widths=(8, 16, 32, 64), fs=(1.0, 2.0), seed0=5000, S=30, epochs=20000):
    out = []
    for w in widths:
        for v in ("F", "D"):
            for f in fs:
                for c0 in coefficient_grid(w):
                    for s0 in coefficient_grid(w):
                        out.append(make_unit("indep_grid", v, w, f, "indep", c0, s0, shortcut_norm="unit", lrs=LRS, S=S, seed0=seed0, epochs=epochs))
    return _snap(out)


EXPERIMENTS = {"lr_budget": lr_budget, "shortcut_scale": shortcut_scale, "shuffled_direction": shuffled_direction,
               "norm_shortcut": norm_shortcut, "indep_grid": indep_grid}
