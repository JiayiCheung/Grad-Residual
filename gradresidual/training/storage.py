"""Files of one experiment: frozen manifest, one npz result per unit, checkpoints per group, exclusive claims."""
import json
import os
import time

import numpy as np

from ..experiment.units import unit_id, unit_key
from ..paths import experiment_dir


def write_atomic(path, data_bytes):
    # a crash leaves either the old file or the complete new one, never a torn file
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def manifest_path(experiment):
    return os.path.join(experiment_dir(experiment), "manifest.json")


def load_manifest(experiment):
    return json.load(open(manifest_path(experiment)))["units"]


def result_path(u):
    return os.path.join(experiment_dir(u["experiment"]), "raw", unit_id(u) + ".npz")


def weights_path(u):
    return os.path.join(experiment_dir(u["experiment"]), "weights", unit_id(u) + ".pt")


def checkpoint_path(experiment, gid):
    return os.path.join(experiment_dir(experiment), "checkpoints", gid + ".pt")


def is_done(u):
    """a unit counts as done only if its result file loads and matches the unit key and the expected shape."""
    p = result_path(u)
    if not os.path.exists(p):
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            n_ev = u["epochs"] // u["eval_every"]
            return meta["key"] == unit_key(u) and z["test"].shape == (len(u["lrs"]) * len(u["orients"]), n_ev, u["S"])
    except Exception:
        return False


def claims_dir(experiment):
    return os.path.join(experiment_dir(experiment), "claims")


def claim(u):
    """exclusive claim by atomic file creation; the file holds the pid of the worker."""
    d = claims_dir(u["experiment"])
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, unit_id(u)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, ("%d %s" % (os.getpid(), time.strftime("%F %T"))).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(u):
    p = os.path.join(claims_dir(u["experiment"]), unit_id(u))
    if os.path.exists(p):
        os.remove(p)


def is_claimed(u):
    return os.path.exists(os.path.join(claims_dir(u["experiment"]), unit_id(u)))


def load_results(experiment):
    """all finished units of an experiment: list of (unit, blocks, arrays)."""
    d = os.path.join(experiment_dir(experiment), "raw")
    out = []
    for name in sorted(os.listdir(d)) if os.path.exists(d) else []:
        if not name.endswith(".npz"):
            continue
        with np.load(os.path.join(d, name), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrs = {k: z[k] for k in z.files if k != "meta"}
        out.append((meta["unit"], meta["blocks"], arrs))
    return out
