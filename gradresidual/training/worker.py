"""Worker loop: repeatedly claim the next group of unfinished units, train it, write the results. Several workers can run on one GPU."""
import gc
import os
import time

import numpy as np
import torch

from ..experiment.units import group_id, group_units, unit_id
from ..paths import experiment_dir
from . import storage
from .trainer import train_group


def run_worker(units, device, max_nets=720, max_cfg=4, log=print):
    n_done = 0
    while True:
        todo = [u for u in units if not storage.is_done(u) and not storage.is_claimed(u)]
        if not todo:
            return n_done
        group = group_units(todo, max_nets, max_cfg)[0]
        mine = [u for u in group if storage.claim(u)]
        if not mine:
            # another worker claimed this group in between
            continue
        try:
            outs = train_group(mine, device, log=log)
        except Exception as ex:
            log("ERROR group %s %r" % (group_id(mine), ex))
            for u in mine:
                storage.release(u)
            raise
        for u, arrs, meta in outs:
            t = arrs["test"][:, -arrs["test"].shape[1] // 4:, :].mean(1)
            fails = (~np.isfinite(t) | (t >= 10)).sum(1)
            log("done %s %s %s w%d f%g %s init=%s init2=%s c=%s crashed seeds per block=%s %.0fs (group of %d)" % (
                unit_id(u), u["experiment"], u["variant"], u["width"], u["f"], u["method"], u["init"], u["init2"], u["c_fixed"],
                fails.tolist(), meta["seconds"], len(outs)))
            storage.release(u)
            n_done += 1
        # hand the cached blocks (and the freed cuda graph pool) back, so a long-running worker does not grow on a shared gpu
        del outs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def file_logger(experiment):
    os.makedirs(experiment_dir(experiment), exist_ok=True)
    f = open(os.path.join(experiment_dir(experiment), "worker_%d.log" % os.getpid()), "a", encoding="utf-8")

    def log(s):
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
        f.flush()
    return log
