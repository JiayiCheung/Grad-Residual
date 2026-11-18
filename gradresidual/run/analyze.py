"""
Selection and analysis of a finished experiment; writes JSON next to the raw results.

    gradresidual_analyze -e lr_budget [--rule drop|penalty] [--freeze]   selection, baseline tallies, interaction, lr x epochs, trained coefficients
    gradresidual_analyze -e shortcut_scale                                crashed seeds and scores per shortcut and learning rate
    gradresidual_analyze -e shuffled_direction                            true vs shuffled g_hat vs Regular, paired by seed

--freeze writes selected.json once (refuses to overwrite); confirmation runs must use a frozen selection.
"""
import argparse
import json
import os

from ..evaluation import summaries as SM
from ..evaluation.scoring import score_table
from ..evaluation.selection import describe, frozen, select
from ..paths import experiment_dir
from ..training.storage import is_done, load_manifest, load_results

BUDGETS = {"lr_budget": (5000, 10000, 20000), "shortcut_scale": (5000,), "shuffled_direction": (5000,),
           "norm_shortcut": (5000, 10000, 20000), "indep_grid": (5000, 10000, 20000),
           "indep_w32": (5000, 10000, 20000), "indep_w32_5k": (5000,), "indep_w64_5k": (5000,)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-e", "--experiment", required=True, choices=sorted(BUDGETS))
    ap.add_argument("--rule", default="drop", choices=("drop", "penalty"), help="how crashed seeds are scored")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()
    d = experiment_dir(a.experiment)
    missing = [u for u in load_manifest(a.experiment) if not is_done(u)]
    if missing and not a.allow_incomplete:
        raise SystemExit("%d units are not finished (use --allow-incomplete for a partial report)" % len(missing))
    tab = score_table(load_results(a.experiment), BUDGETS[a.experiment], a.rule)

    if a.experiment == "lr_budget":
        win = select(tab)
        T = max(BUDGETS[a.experiment])
        out = dict(rule=a.rule, winners=[dict(variant=k[0], width=k[1], f=k[2], T=k[3], family=k[4], **describe(c, v)) for k, (c, v) in sorted(win.items(), key=str)],
                   baselines=SM.baseline_table(win), best_per_scenario=SM.best_per_scenario(win), interaction=SM.interaction(win),
                   lr_epoch=SM.lr_epoch(tab, win, T, T // 4), trained_c=SM.trained_coefficients(tab, "grad_only", lrs=(0.03125, 0.0078125)))
        if a.freeze:
            p = os.path.join(d, "selected.json")
            if os.path.exists(p):
                raise SystemExit("selected.json exists (frozen)")
            json.dump(frozen(win), open(p, "w"), indent=1)
            print("selected.json written")
    elif a.experiment == "shortcut_scale":
        out = dict(rule=a.rule, crash_counts=SM.crash_counts(tab))
    else:
        out = dict(rule=a.rule, contrasts=SM.shuffled_contrasts(tab))
    p = os.path.join(d, "analysis_%s.json" % a.rule)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=float)
    print("analysis -> %s" % p)


if __name__ == "__main__":
    main()
