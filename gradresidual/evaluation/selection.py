"""
Every method at its own best: per (variant, width, f, budget, family) the configuration (coefficients x learning rate) with the lowest mean score.

Families: regular, standard_fixed, standard_train, grad_res (Grad Residual: c trainable or c fixed), addition, convex, grad_mag, plus the named
comparison grad_fixed1 (Grad Residual with c fixed at 1, learning rate tuned). Regular is trained under F only and shared with D.
"""
import hashlib
import json

import numpy as np

FAMILY = {"regular": "regular", "standard_fixed": "standard_fixed", "standard_train": "standard_train", "grad_only": "grad_res", "grad_fixed": "grad_res",
          "addition": "addition", "convex": "convex", "grad_mag": "grad_mag"}


def families(cfg):
    out = [FAMILY[cfg[0]]]
    if cfg[0] == "grad_fixed" and cfg[3] == 1.0:
        out.append("grad_fixed1")
    return out


def select(tab):
    """{(variant, width, f, T, family): (cfg, entry)}; a configuration needs at least 2 kept seeds to compete."""
    win = {}
    for key, cfgs in tab.items():
        for cfg, v in cfgs.items():
            if v["kept"] < 2:
                continue
            s = np.nanmean(v["over"])
            for fam in families(cfg):
                k = key + (fam,)
                if k not in win or s < np.nanmean(win[k][1]["over"]):
                    win[k] = (cfg, v)
    for k in list(win):
        if k[0] == "F" and k[4] == "regular":
            win[("D",) + k[1:]] = win[k]
    return win


def describe(cfg, v):
    m, i, i2, c, norm, shuf, lr = cfg
    fin = v["fin"]
    return dict(method=m, init=i, init2=i2, c_fixed=c, shortcut_norm=norm, shuffle=shuf, lr=lr,
                score=float(np.nanmean(v["over"])), score_sd=float(np.nanstd(v["over"], ddof=1)),
                hard=float(np.nanmean(v["hard"])), hard_sd=float(np.nanstd(v["hard"], ddof=1)), crashes=v["crashes"], kept=v["kept"],
                trained=None if fin is None or not len(fin) else dict(median=float(np.median(fin)), p10=float(np.percentile(fin, 10)),
                                                                     p90=float(np.percentile(fin, 90)), n=int(len(fin))))


def frozen(win):
    """the selection as a sorted list with its hash (written once, before any confirmation run)."""
    sel = [dict(variant=k[0], width=k[1], f=k[2], budget=k[3], family=k[4], **describe(c, v)) for k, (c, v) in sorted(win.items(), key=lambda kv: json.dumps(kv[0]))]
    h = hashlib.sha256(json.dumps([{k: s[k] for k in ("variant", "width", "f", "budget", "family", "method", "init", "init2", "c_fixed", "lr")} for s in sel],
                                  sort_keys=True).encode()).hexdigest()
    return dict(hash=h, selected=sel)
