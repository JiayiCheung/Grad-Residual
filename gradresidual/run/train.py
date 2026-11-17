"""
Manifest and training of a registered experiment.

    gradresidual_train -e lr_budget --make-manifest       freeze the unit list (refuses to overwrite)
    gradresidual_train -e lr_budget                        progress
    gradresidual_train -e lr_budget --worker               claim and train groups until none is left (start several for one GPU)
    gradresidual_train -e lr_budget --worker --widths 8,16 --max-nets 360 --max-cfg 2
    gradresidual_train -e lr_budget --release-claims       only when NO worker is running (e.g. after a reboot)
"""
import argparse
import json
import os

import torch

from ..experiment.registry import EXPERIMENTS
from ..paths import experiment_dir
from ..training import storage
from ..training.trainer import CODE_HASH
from ..training.worker import file_logger, run_worker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiment", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--make-manifest", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--widths", default=None, help="comma-separated widths this worker takes (default: all)")
    ap.add_argument("--max-nets", type=int, default=720, help="networks per process at width 32 (scaled by 32 / width)")
    ap.add_argument("--max-cfg", type=int, default=4, help="configurations stacked per process")
    ap.add_argument("--release-claims", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.make_manifest:
        p = storage.manifest_path(a.experiment)
        if os.path.exists(p):
            raise SystemExit("manifest exists (frozen): %s" % p)
        units = EXPERIMENTS[a.experiment]()
        os.makedirs(experiment_dir(a.experiment), exist_ok=True)
        json.dump(dict(code=CODE_HASH, units=units), open(p, "w"), indent=0)
        print("manifest: %d units -> %s" % (len(units), p))
        return

    units = storage.load_manifest(a.experiment)
    if a.widths:
        keep = {int(w) for w in a.widths.split(",")}
        units = [u for u in units if u["width"] in keep]
    if a.release_claims:
        d = storage.claims_dir(a.experiment)
        for f in os.listdir(d) if os.path.exists(d) else []:
            os.remove(os.path.join(d, f))
        print("claims released")
        return
    print("%s: %d units, %d done" % (a.experiment, len(units), sum(storage.is_done(u) for u in units)), flush=True)
    if a.worker:
        run_worker(units, a.device, a.max_nets, a.max_cfg, log=file_logger(a.experiment))


if __name__ == "__main__":
    main()
