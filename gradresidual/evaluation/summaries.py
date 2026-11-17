"""The comparisons behind each conclusion, computed from a score table and its selection."""
import collections

import numpy as np

from .statistics import common_mean, paired, tally

GRADIENT_FAMILIES = ("grad_res", "addition", "convex", "grad_mag")
BASELINES = ("regular", "standard_fixed")


def versus(win, fam, base, key="over", where=None):
    """paired differences fam - base over all scenarios (each family at its own best), with the two raw means."""
    rows = []
    for k, (cfg, v) in win.items():
        if k[4] != fam or (k[:4] + (base,)) not in win or (where and not where(k)):
            continue
        b = win[k[:4] + (base,)][1]
        rows.append(dict(variant=k[0], width=k[1], f=k[2], T=k[3], diff=paired(v[key], b[key]),
                         mean=common_mean(v[key], b[key]), base_mean=common_mean(b[key], v[key])))
    return rows


def baseline_table(win):
    """for every gradient family and Standard Residual: tallies against both baselines, overall and on the hard half."""
    out = {}
    for fam in GRADIENT_FAMILIES + ("standard_fixed", "standard_train", "grad_fixed1"):
        for base in BASELINES:
            if fam == base:
                continue
            for key in ("over", "hard"):
                out["%s|%s|%s" % (fam, base, key)] = tally([r["diff"] for r in versus(win, fam, base, key)])
    return out


def best_per_scenario(win, fams=("regular", "standard_fixed", "standard_train", "grad_res", "addition", "convex", "grad_mag")):
    c = collections.Counter()
    for key in {k[:4] for k in win}:
        sc = {f: np.nanmean(win[key + (f,)][1]["over"]) for f in fams if key + (f,) in win}
        c[min(sc, key=sc.get)] += 1
    return dict(c)


def interaction(win, fam="grad_res", base="regular"):
    """I = G(f = 2) - G(f = 1) on the hard half, G = fam - base, each at its own best, per (variant, width, budget)."""
    out = []
    for (v, w, f, T, fm) in win:
        if fm != fam or f != 1.0 or (v, w, 2.0, T, fam) not in win:
            continue
        g1 = win[(v, w, 1.0, T, fam)][1]["hard"] - win[(v, w, 1.0, T, base)][1]["hard"]
        g2 = win[(v, w, 2.0, T, fam)][1]["hard"] - win[(v, w, 2.0, T, base)][1]["hard"]
        out.append(dict(variant=v, width=w, T=T, G1=float(np.nanmean(g1)), G2=float(np.nanmean(g2)), I=paired(g2, g1)))
    return out


def lr_epoch(tab, win, T_long, T_short, fams=(("regular", None), ("grad_fixed", 1.0))):
    """(4 eta, T_short) against (eta, T_long) at each method's best eta for T_long; T_short = T_long / 4 keeps lr x steps equal."""
    out = []
    for (v, w, f, T, fam), (cfg, slow) in win.items():
        if T != T_long:
            continue
        for meth, cf in fams:
            if cfg[0] != meth or cfg[3] != cf or (fam == "grad_res" and meth == "grad_fixed"):
                continue
            fast_cfg = cfg[:-1] + (4 * cfg[-1],)
            fast = tab.get((v, w, f, T_short), {}).get(fast_cfg)
            if fast is None:
                continue
            out.append(dict(variant=v, width=w, f=f, method=meth, lr=cfg[-1], slow=common_mean(slow["over"], fast["over"]),
                            fast=common_mean(fast["over"], slow["over"]), diff=paired(fast["over"], slow["over"])))
    return out


def trained_coefficients(tab, method="grad_only", T=None, lrs=None):
    """trained c (or s, alpha) at the end of T, pooled over variants, f and orientations, per (width, start value, lr)."""
    pool = collections.defaultdict(list)
    for (v, w, f, TT), cfgs in tab.items():
        if T is not None and TT != T:
            continue
        for cfg, e in cfgs.items():
            if cfg[0] != method or e["fin"] is None or (lrs and cfg[-1] not in lrs):
                continue
            pool[(w, cfg[1], cfg[-1], TT)].append(e["fin"])
    out = []
    for (w, c0, lr, TT), v in sorted(pool.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2], kv[0][3])):
        c = np.concatenate(v)
        out.append(dict(width=w, start=c0, lr=lr, T=TT, n=int(len(c)), median=float(np.median(c)), abs_median=float(np.median(np.abs(c))),
                        negative=float((c < 0).mean()), abs_over_sqrt_width=float(np.median(np.abs(c)) / np.sqrt(w))))
    return out


def crash_counts(tab):
    """crashed seeds per configuration and learning rate (both rules give the same counts)."""
    out = []
    for (v, w, f, T), cfgs in sorted(tab.items()):
        for cfg, e in cfgs.items():
            out.append(dict(variant=v, width=w, f=f, T=T, method=cfg[0], s=cfg[1], shortcut_norm=cfg[4], lr=cfg[-1], crashes=e["crashes"],
                            seeds=len(e["over"]), score=float(np.nanmean(e["over"]))))
    return out


def shuffled_contrasts(tab):
    """per (variant, width, orientation-averaged): shuffled - true, shuffled - regular, true - regular (paired by seed)."""
    out = []
    for (v, w, f, T), cfgs in sorted(tab.items()):
        true = [e for c, e in cfgs.items() if c[0] == "grad_fixed" and not c[5]]
        shuf = [e for c, e in cfgs.items() if c[0] == "grad_fixed" and c[5]]
        reg = tab.get(("F", w, f, T), {})
        reg = [e for c, e in reg.items() if c[0] == "regular"]
        if not (true and shuf and reg):
            continue
        t, s, r = true[0]["over"], shuf[0]["over"], reg[0]["over"]
        out.append(dict(variant=v, width=w, T=T, regular=float(np.nanmean(r)), true=float(np.nanmean(t)), shuffled=float(np.nanmean(s)),
                        shuffled_minus_true=paired(s, t), shuffled_minus_regular=paired(s, r), true_minus_regular=paired(t, r)))
    return out
