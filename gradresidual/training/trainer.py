"""
Stacked training of one group (several configurations of one method): n_configs x n_lrs x n_orients x S networks on the leading axis, plain SGD
(momentum 0, constant learning rate per network, batch 512, MSE). Data, initial weights and batch permutations are drawn once for S seeds and repeated over
all blocks, so every block is paired with every other block. On CUDA one epoch is captured as a CUDA graph and replayed; the permutation is drawn
eagerly each epoch into a static buffer. Checkpoints every ckpt_every epochs make the run resumable after a hard kill.
"""
import hashlib
import io
import json
import os
import time

import numpy as np
import torch

from ..data.sine import N_TRAIN, hard_mask, make_data
from ..experiment.units import GROUP_FIELDS, blocks, group_id, group_key, is_trainable, unit_key
from ..network.stacked_net import VARIANTS, StackedNet, digest, draw_weights
from .storage import checkpoint_path, result_path, weights_path, write_atomic

BATCH_SIZE = 512
# three independent random streams per seed set: data, initial weights, batch order
DATA_OFFSET, INIT_OFFSET, SHUF_OFFSET = 12345, 7_000_000, 9_000_000
GHAT_SHUFFLE_OFFSET = 31337
CODE_HASH = hashlib.sha1(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]


def make_streams(seed0, device):
    mk = lambda off: torch.Generator(device=device).manual_seed(off + seed0)
    return mk(DATA_OFFSET), mk(INIT_OFFSET), mk(SHUF_OFFSET)


def stack_data(u, device):
    """data of the S seeds for every orientation (same data stream each time), repeated in block order (lr outer, orientation inner)."""
    S = u["S"]
    per = {}
    for o in u["orients"]:
        gd, _, _ = make_streams(u["seed0"], device)
        per[o] = make_data(gd, S, device, u["f"], o)
    fp = digest(per[u["orients"][0]][0], per[u["orients"][0]][1])
    cat = lambda i: torch.cat([per[o][i] for _ in u["lrs"] for o in u["orients"]], 0)
    return [cat(i) for i in range(6)], fp


WEIGHT_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3", "Wo", "bo", "s_fix", "alpha", "beta", "mask")


def save_weights(net, units, ep, fp, per_unit, S):
    """one .pt per unit: {epoch: {name: tensor (blocks * S, ...)}} plus the layout needed to read it; rewritten atomically at every snapshot."""
    for k, u in enumerate(units):
        sl = slice(k * per_unit * S, (k + 1) * per_unit * S)
        p = weights_path(u)
        rec = torch.load(p, weights_only=False) if os.path.exists(p) else dict(
            unit=u, unit_key=unit_key(u), blocks=blocks(u), seeds=S, seed0=u["seed0"], fingerprint=dict(fp), code=CODE_HASH,
            layout="rows = blocks x seeds (block order: learning rate outer, orientation inner; seed index inner)", snapshots={})
        rec["snapshots"][ep] = {n: getattr(net, n)[sl].detach().cpu().clone() for n in WEIGHT_NAMES if getattr(net, n, None) is not None}
        buf = io.BytesIO()
        torch.save(rec, buf)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        write_atomic(p, buf.getvalue())


def train_group(units, device, ckpt_every=2000, use_graph=None, log=None, save_at=None, progress=None):
    """trains the units of one group and writes one npz result per unit (and the weights at the epochs save_at); returns [(unit, arrays, meta)].
    progress(epoch, epochs, seconds, val, test), if given, is called at every evaluation with the mean errors over the finite networks."""
    u0 = units[0]
    for u in units:
        assert all(u[k] == u0[k] for k in GROUP_FIELDS), "units of a group must agree on %s" % (GROUP_FIELDS,)
    S, epochs, ev = u0["S"], u0["epochs"], u0["eval_every"]
    save_at = set(u0.get("save_weights", ()) if save_at is None else save_at)
    nl, no = len(u0["lrs"]), len(u0["orients"])
    cuda = str(device).startswith("cuda")
    # the shuffling generator is not registered with the graph, so shuffled groups run eagerly
    use_graph = (cuda and not u0["shuffle"]) if use_graph is None else use_graph
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    (X1, Y1, Xv1, Yv1, Xt1, Yt1), fp_data = stack_data(u0, device)
    rc = lambda t: t.repeat(len(units), 1, 1)
    X, Y, Xv, Yv, Xt, Yt = rc(X1), rc(Y1), rc(Xv1), rc(Yv1), rc(Xt1), rc(Yt1)
    del X1, Y1, Xv1, Yv1, Xt1, Yt1

    _, gi, gs = make_streams(u0["seed0"], device)
    weights = draw_weights(gi, S, u0["width"], grad_mag=u0["method"] == "grad_mag", device=device)
    fp_init = digest(*[weights[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3", "Wo", "bo")])
    sgen = None
    if u0["shuffle"]:
        sgen = torch.Generator(device=device).manual_seed(GHAT_SHUFFLE_OFFSET + u0["seed0"])
    net = StackedNet(units, weights, nl * no, S, VARIANTS[u0["variant"]], device, u0["shortcut_norm"], sgen)
    nb = net.nb
    lr_vec = torch.tensor([b["lr"] for _ in units for b in blocks(u0)], device=device, dtype=torch.float32).repeat_interleave(S).view(-1, 1, 1)
    hm = torch.stack([hard_mask(Xt, b["orient"]) for _ in units for b in blocks(u0)]).repeat_interleave(S, 0).float()

    curves = {k: [] for k in ("val", "test", "test_hard", "test_easy", "scal", "scal_s")}
    fp = dict(data=fp_data, init=fp_init)
    start, elapsed = 1, 0.0
    cp = checkpoint_path(u0["experiment"], group_id(units))
    params = net.params()
    if os.path.exists(cp):
        st = torch.load(cp, map_location=device, weights_only=False)
        if st["key"] == group_key(units) and st["code"] == CODE_HASH and st["fp_init"] == fp["init"] and st["fp_data"] == fp["data"]:
            with torch.no_grad():
                for p, q in zip(params, st["params"]):
                    p.copy_(q)
            gs.set_state(st["gs"].cpu())
            if sgen is not None and st.get("sgen") is not None:
                sgen.set_state(st["sgen"].cpu())
            curves = st["curves"]
            fp.update(st["fp_perm"])
            start, elapsed = st["epoch"] + 1, st["elapsed"]
            if log:
                log("resume group %s from epoch %d" % (group_id(units), st["epoch"]))
        elif log:
            log("incompatible checkpoint ignored %s" % group_id(units))

    batches = [(i, min(i + BATCH_SIZE, N_TRAIN)) for i in range(0, N_TRAIN, BATCH_SIZE)]
    perm_buf = torch.zeros(nb * S, N_TRAIN, dtype=torch.long, device=device)

    def epoch_body():
        for (i, j) in batches:
            idx = perm_buf[:, i:j].unsqueeze(-1)
            xb, yb = X.gather(1, idx), Y.gather(1, idx)
            loss = ((net(xb) - yb) ** 2).mean(dim=(1, 2)).sum()
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, gr in zip(params, grads):
                    # fixed coefficients that share the tensor with trainable ones are frozen by the mask
                    if p is net.alpha and net.mask is not None:
                        gr = gr * net.mask
                    p.sub_(gr * lr_vec)

    graph, warm = None, 0
    side = torch.cuda.Stream() if use_graph else None
    t0 = time.time()
    for ep in range(start, epochs + 1):
        perm = torch.argsort(torch.rand(S, N_TRAIN, generator=gs, device=device), dim=1)
        if ep <= 2:
            fp["perm%d" % ep] = digest(perm)
        perm_buf.copy_(perm.repeat(nb, 1))
        if not use_graph:
            epoch_body()
        elif graph is None and warm < 2:
            # graph capture requires warm-up iterations on a side stream; these are real training epochs
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                epoch_body()
            torch.cuda.current_stream().wait_stream(side)
            warm += 1
            if warm == 2:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    epoch_body()  # recorded, not executed
        else:
            graph.replay()
        if ep % ev == 0:
            sync()
            curves["val"].append(((net(Xv).detach() - Yv) ** 2).mean(dim=(1, 2)).cpu().numpy().astype(np.float32))
            e = ((net(Xt).detach() - Yt) ** 2)[..., 0]
            curves["test"].append(e.mean(1).cpu().numpy().astype(np.float32))
            curves["test_hard"].append(((e * hm).sum(1) / hm.sum(1)).cpu().numpy().astype(np.float32))
            curves["test_easy"].append(((e * (1 - hm)).sum(1) / (1 - hm).sum(1)).cpu().numpy().astype(np.float32))
            curves["scal"].append(net.alpha.detach().view(-1).cpu().numpy().astype(np.float32) if net.alpha is not None else np.zeros(0, np.float32))
            curves.setdefault("scal_s", []).append(net.beta.detach().view(-1).cpu().numpy().astype(np.float32) if net.beta is not None else np.zeros(0, np.float32))
            if progress is not None:
                v, t = curves["val"][-1], curves["test"][-1]
                progress(ep, epochs, elapsed + time.time() - t0, float(np.nanmean(np.where(np.isfinite(v), v, np.nan))), float(np.nanmean(np.where(np.isfinite(t), t, np.nan))))
        if ep in save_at:
            sync()
            save_weights(net, units, ep, fp, nl * no, S)
        if ep % ckpt_every == 0 and ep < epochs:
            sync()
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            buf = io.BytesIO()
            torch.save(dict(key=group_key(units), code=CODE_HASH, epoch=ep, params=[p.detach().clone() for p in params], gs=gs.get_state(),
                            sgen=sgen.get_state() if sgen is not None else None, curves=curves, fp_init=fp["init"], fp_data=fp["data"],
                            fp_perm={k: v for k, v in fp.items() if k.startswith("perm")}, elapsed=elapsed + time.time() - t0), buf)
            write_atomic(cp, buf.getvalue())
    sync()
    elapsed += time.time() - t0

    # split the stacked curves back into the units: arrays of shape (blocks, n_eval, S)
    per_unit = nl * no
    outs = []
    for k, u in enumerate(units):
        sl = slice(k * per_unit * S, (k + 1) * per_unit * S)
        shape = lambda L: np.stack(L, 0)[:, sl].reshape(len(L), per_unit, S).transpose(1, 0, 2)
        arrs = {key: shape(curves[key]) for key in ("val", "test", "test_hard", "test_easy")}
        if is_trainable(u):
            arrs["scal"] = shape(curves["scal"])
        if net.beta is not None:
            arrs["scal_s"] = shape(curves["scal_s"])
        meta = dict(key=unit_key(u), unit=u, blocks=blocks(u), fingerprint=fp, code=CODE_HASH, group=group_id(units), group_size=len(units),
                    cuda_graph=bool(use_graph), torch=torch.__version__, device=str(device), seconds=elapsed / len(units), resumed_from=start - 1)
        buf = io.BytesIO()
        np.savez_compressed(buf, meta=np.array(json.dumps(meta)), **arrs)
        os.makedirs(os.path.dirname(result_path(u)), exist_ok=True)
        write_atomic(result_path(u), buf.getvalue())
        outs.append((u, arrs, meta))
    if os.path.exists(cp):
        os.remove(cp)
    return outs
