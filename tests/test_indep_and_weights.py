import numpy as np
import pytest
import torch

import gradresidual.paths as paths
from gradresidual.training.trainer import stack_data
from gradresidual.experiment.units import make_unit
from gradresidual.network.stacked_net import StackedNet, draw_weights


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_ROOT", str(tmp_path))
    return tmp_path


def test_indep_at_start_equals_addition_with_the_same_coefficients():
    w = draw_weights(torch.Generator().manual_seed(3), 4, 8)
    x = torch.linspace(-3, 3, 50).view(1, -1, 1).repeat(4, 1, 1)
    for span in (1, 2):
        cfg_i = [dict(method="indep", init=0.3, init2=0.7, c_fixed=None)]
        cfg_a = [dict(method="addition", init=0.3, init2=0.7, c_fixed=None)]
        yi = StackedNet(cfg_i, w, 1, 4, span, "cpu", "unit")(x)
        ya = StackedNet(cfg_a, w, 1, 4, span, "cpu", "unit")(x)
        assert torch.allclose(yi, ya, atol=1e-6)


def _indep_units(epochs=6):
    out = [make_unit("t", "D", 8, 1.0, "indep", c, s, shortcut_norm="unit", lrs=(0.03125, 0.0078125), S=2, seed0=11, epochs=epochs, eval_every=2)
           for (c, s) in ((0.1, 1.0), (1.0, 0.01))]
    for u in out:
        u["save_weights"] = [2, 6]
    return out


def test_indep_trains_both_coefficients_and_saves_weights(results_root):
    from gradresidual.training.storage import weights_path
    from gradresidual.training.trainer import train_group
    outs = train_group(_indep_units(), "cpu", use_graph=False)
    for u, arrs, meta in outs:
        assert arrs["scal"].shape == arrs["scal_s"].shape == (4, 3, 2)
        assert not np.allclose(arrs["scal"][:, -1], u["init"]) and not np.allclose(arrs["scal_s"][:, -1], u["init2"])
        rec = torch.load(weights_path(u), weights_only=False)
        assert sorted(rec["snapshots"]) == [2, 6] and rec["seeds"] == 2 and len(rec["blocks"]) == 4
        snap = rec["snapshots"][6]
        assert snap["W2"].shape == (8, 8, 8) and snap["beta"].shape == (8, 1, 1)
        # the saved weights reproduce the recorded final test error, block by block
        net = StackedNet([u], {k: snap[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3", "Wo", "bo")}, 1, 8, 1, "cpu", "unit")
        net.alpha.data.copy_(snap["alpha"])
        net.beta.data.copy_(snap["beta"])
        (_, _, _, _, Xt, Yt), _ = stack_data(u, "cpu")
        err = ((net(Xt).detach() - Yt) ** 2).mean(dim=(1, 2)).numpy().reshape(4, 2)
        assert np.allclose(err, arrs["test"][:, -1], rtol=1e-4)


def test_resume_keeps_weight_snapshots_identical(results_root, monkeypatch):
    from gradresidual.training import trainer
    from gradresidual.training.storage import weights_path
    full = trainer.train_group(_indep_units(), "cpu", use_graph=False)
    ref = {u["init"]: torch.load(weights_path(u), weights_only=False)["snapshots"][6]["W3"] for u in _indep_units()}
    import shutil
    shutil.rmtree(results_root / "t")

    real = trainer.write_atomic
    def die_after_checkpoint(path, data):
        real(path, data)
        if "checkpoints" in path:
            raise KeyboardInterrupt
    monkeypatch.setattr(trainer, "write_atomic", die_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        trainer.train_group(_indep_units(), "cpu", use_graph=False, ckpt_every=4)
    monkeypatch.setattr(trainer, "write_atomic", real)
    resumed = trainer.train_group(_indep_units(), "cpu", use_graph=False, ckpt_every=4)
    for a, b in zip(full, resumed):
        for key in ("test", "scal", "scal_s"):
            assert np.array_equal(a[1][key], b[1][key])
        rec = torch.load(weights_path(b[0]), weights_only=False)
        assert sorted(rec["snapshots"]) == [2, 6] and torch.equal(rec["snapshots"][6]["W3"], ref[b[0]["init"]])


def test_progress_is_reported_at_every_evaluation_and_does_not_change_results(results_root):
    from gradresidual.training.trainer import train_group
    calls = []
    a = train_group(_indep_units(), "cpu", use_graph=False, progress=lambda *x: calls.append(x))
    assert [c[0] for c in calls] == [2, 4, 6] and all(c[1] == 6 for c in calls)
    import shutil
    shutil.rmtree(results_root / "t")
    b = train_group(_indep_units(), "cpu", use_graph=False)
    for x, y in zip(a, b):
        assert np.array_equal(x[1]["test"], y[1]["test"])
