import importlib

import numpy as np
import pytest

import gradresidual.paths as paths
from gradresidual.evaluation.scoring import score_table
from gradresidual.evaluation.selection import select
from gradresidual.evaluation.statistics import paired, tally
from gradresidual.experiment.units import make_unit


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_ROOT", str(tmp_path))
    return tmp_path


def _units(epochs=4):
    return [make_unit("t", "F", 8, 1.0, "grad_only", c, lrs=(0.03125, 0.0078125), S=2, seed0=11, epochs=epochs, eval_every=2) for c in (0.1, 1.0)]


def test_group_equals_separate_runs(results_root):
    from gradresidual.training.trainer import train_group
    both = train_group(_units(), "cpu", use_graph=False)
    for k, u in enumerate(_units()):
        alone = train_group([u], "cpu", use_graph=False)[0]
        assert np.allclose(both[k][1]["test"], alone[1]["test"], rtol=1e-5, atol=1e-7)


def test_resume_after_a_hard_stop_is_identical(results_root, monkeypatch):
    from gradresidual.training import trainer
    full = trainer.train_group(_units(), "cpu", use_graph=False)

    real = trainer.write_atomic
    def write_then_die(path, data):
        real(path, data)
        if path.endswith(".pt"):
            raise KeyboardInterrupt
    monkeypatch.setattr(trainer, "write_atomic", write_then_die)
    with pytest.raises(KeyboardInterrupt):
        trainer.train_group(_units(), "cpu", use_graph=False, ckpt_every=2)
    monkeypatch.setattr(trainer, "write_atomic", real)
    resumed = trainer.train_group(_units(), "cpu", use_graph=False, ckpt_every=2)
    for a, b in zip(full, resumed):
        for key in ("test", "scal"):
            assert np.array_equal(a[1][key], b[1][key])
    assert resumed[0][2]["resumed_from"] == 2


def _fake_results():
    u = make_unit("t", "F", 8, 1.0, "regular", lrs=(0.1, 0.01), orients=("right", "left"), S=4, epochs=40, eval_every=10)
    test = np.full((4, 4, 4), 0.5, np.float32)
    test[0:2, :, 0] = np.inf  # lr 0.1: seed 0 crashes in both orientations
    test[2:4] = 0.3
    blocks = [dict(lr=lr, orient=o) for lr in (0.1, 0.01) for o in ("right", "left")]
    return [(u, blocks, dict(test=test, test_hard=test.copy()))]


def test_crash_rules():
    tab = score_table(_fake_results(), (40,), "drop")
    drop = tab[("F", 8, 1.0, 40)]
    pen = score_table(_fake_results(), (40,), "penalty")[("F", 8, 1.0, 40)]
    k = [c for c in drop if c[-1] == 0.1][0]
    assert drop[k]["kept"] == 3 and drop[k]["crashes"] == 2 and np.isnan(drop[k]["over"][0])
    assert pen[k]["over"][0] == 10.0
    win = select(tab)
    assert win[("F", 8, 1.0, 40, "regular")][0][-1] == 0.01 and ("D", 8, 1.0, 40, "regular") in win


def test_paired_statistics():
    a, b = np.array([1.0, 2.0, 3.0, np.nan]), np.array([0.5, 1.5, 2.6, 1.0])
    p = paired(a, b)
    assert p["n"] == 3 and abs(p["d"] - 0.4666667) < 1e-6 and p["lo"] < p["d"] < p["hi"]
    t = tally([p, dict(d=-1.0, lo=-2.0, hi=-0.5)])
    assert t == dict(n=2, lower=1, better=1, worse=1, median=t["median"])
