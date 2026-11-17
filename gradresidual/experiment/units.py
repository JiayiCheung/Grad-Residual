"""
A unit is one method configuration (method, coefficients) of one experiment cell (variant, width, f), trained for several learning rates x orientations
x S seeds. Units are the claim / result granularity; units that differ only in their coefficients are stacked into one process (a group).
"""
import hashlib
import json

UNIT_FIELDS = ("experiment", "variant", "width", "f", "method", "init", "init2", "c_fixed", "shortcut_norm", "shuffle", "lrs", "orients", "S", "seed0", "epochs", "eval_every")
# fields that must agree inside a group (the coefficients init / init2 / c_fixed may differ)
GROUP_FIELDS = ("experiment", "variant", "width", "f", "method", "shortcut_norm", "shuffle", "lrs", "orients", "S", "seed0", "epochs", "eval_every")


def make_unit(experiment, variant, width, f, method, init=None, init2=None, c_fixed=None, shortcut_norm="none", shuffle=False,
              lrs=(), orients=("right", "left"), S=30, seed0=0, epochs=20000, eval_every=100):
    return dict(experiment=experiment, variant=variant, width=width, f=float(f), method=method, init=init, init2=init2, c_fixed=c_fixed,
                shortcut_norm=shortcut_norm, shuffle=bool(shuffle), lrs=[float(x) for x in lrs], orients=list(orients), S=S, seed0=seed0,
                epochs=epochs, eval_every=eval_every)


def _r9(x):
    return None if x is None else round(float(x), 9)


def unit_key(u):
    v = [u["experiment"], u["variant"], u["width"], _r9(u["f"]), u["method"], _r9(u["init"]), _r9(u["init2"]), _r9(u["c_fixed"]),
         u["shortcut_norm"], bool(u["shuffle"]), [_r9(x) for x in u["lrs"]], list(u["orients"]), u["S"], u["seed0"], u["epochs"], u["eval_every"]]
    return json.dumps(v)


def unit_id(u):
    return hashlib.sha1(unit_key(u).encode()).hexdigest()[:20]


def blocks(u):
    """block order inside a unit: learning rate outer, orientation inner."""
    return [dict(lr=lr, orient=o) for lr in u["lrs"] for o in u["orients"]]


def is_trainable(u):
    m = u["method"]
    return m in ("standard_train", "grad_only", "convex") or (m == "addition" and u["init"] is not None)


def group_units(units, max_nets=720, max_cfg=4):
    """consecutive units that agree on GROUP_FIELDS form groups, capped at max_nets * 32 / width networks and max_cfg configurations."""
    key = lambda u: tuple(json.dumps(u[k]) for k in GROUP_FIELDS)
    out, cur = [], []
    for u in units:
        nets = len(u["lrs"]) * len(u["orients"]) * u["S"]
        cap = max(1, min(max_cfg, int(max_nets * 32 / u["width"]) // nets))
        if cur and (key(u) != key(cur[0]) or len(cur) >= cap):
            out.append(cur)
            cur = []
        cur.append(u)
    if cur:
        out.append(cur)
    return out


def group_key(units):
    return json.dumps([unit_key(u) for u in units])


def group_id(units):
    return hashlib.sha1(group_key(units).encode()).hexdigest()[:20]
