import numpy as np
from scipy import stats


def paired(a, b):
    """mean of a - b over the seeds where both are finite, with a 95% t interval."""
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return dict(d=float("nan"), lo=float("nan"), hi=float("nan"), n=n)
    m = d.mean()
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(d=float(m), lo=float(m - h), hi=float(m + h), n=n)


def common_mean(a, b):
    """mean of a over the seeds kept in both a and b, so that a pair of means matches its paired difference."""
    k = np.isfinite(a) & np.isfinite(b)
    return float(a[k].mean()) if k.any() else float("nan")


def tally(diffs):
    """counts over scenarios: lower mean, significantly lower (interval < 0), significantly higher (interval > 0), median difference."""
    ok = [p for p in diffs if p["d"] == p["d"]]
    return dict(n=len(ok), lower=sum(p["d"] < 0 for p in ok), better=sum(p["hi"] < 0 for p in ok), worse=sum(p["lo"] > 0 for p in ok),
                median=float(np.median([p["d"] for p in ok])) if ok else float("nan"))
