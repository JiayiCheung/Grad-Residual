"""
Per-seed scores read from the saved curves.

score at budget T = mean test MSE over the last 25% of the evaluations up to T (the curve is a prefix, so 5000 / 10000 / 20000 come from one run).
A seed has crashed at T if its tail mean is non-finite or >= 10 in either orientation. Two rules:
    "drop"     crashed seeds are removed (comparisons then use the seeds kept on both sides)
    "penalty"  a crashed seed scores 10
Orientations are averaged within each seed.
"""
import numpy as np

CRASH = 10.0


def tail(curve, T, eval_every):
    n = T // eval_every
    k = max(1, n // 4)
    return curve[n - k:n].mean(0)


def score_table(results, budgets, rule="drop"):
    """{(variant, width, f, T): {(method, init, init2, c_fixed, shortcut_norm, shuffle, lr): entry}}; entry holds per-seed vectors
    over (overall), hard (hard half), the crash count over both orientations, the number of kept seeds and the trained coefficient at T."""
    assert rule in ("drop", "penalty")
    tab = {}
    for u, blks, arrs in results:
        by_lr = {}
        for b, blk in enumerate(blks):
            by_lr.setdefault(blk["lr"], {})[blk["orient"]] = b
        for lr, ob in by_lr.items():
            if set(ob) != set(u["orients"]):
                continue
            for T in budgets:
                if T > u["epochs"]:
                    continue
                ov, hd, bad = [], [], []
                for o in u["orients"]:
                    t = tail(arrs["test"][ob[o]].astype(np.float64), T, u["eval_every"])
                    crashed = ~np.isfinite(t) | (t >= CRASH)
                    ov.append(np.where(crashed, CRASH, t))
                    hd.append(np.where(crashed, CRASH, tail(arrs["test_hard"][ob[o]].astype(np.float64), T, u["eval_every"])))
                    bad.append(crashed)
                over, hard = np.mean(ov, 0), np.mean(hd, 0)
                anybad = np.any(bad, 0)
                if rule == "drop":
                    over, hard = np.where(anybad, np.nan, over), np.where(anybad, np.nan, hard)
                fin = None
                if "scal" in arrs:
                    keep = ~anybad
                    fin = np.concatenate([arrs["scal"][ob[o]][T // u["eval_every"] - 1][keep].astype(np.float64) for o in u["orients"]])
                    fin = fin[np.isfinite(fin)]
                cfg = (u["method"], u["init"], u["init2"], u["c_fixed"], u.get("shortcut_norm", "none"), bool(u.get("shuffle", False)), lr)
                tab.setdefault((u["variant"], u["width"], u["f"], T), {})[cfg] = dict(
                    over=over, hard=hard, crashes=int(np.sum(bad)), kept=int(np.isfinite(over).sum()), fin=fin)
    return tab
