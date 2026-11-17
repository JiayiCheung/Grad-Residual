import torch

from gradresidual.data.sine import N_TEST, hard_mask, make_data, target
from gradresidual.experiment.registry import EXPERIMENTS, method_configs
from gradresidual.experiment.units import group_units, make_unit
from gradresidual.network.stacked_net import StackedNet, draw_weights, shuffle_nonzero


def test_mirror_target():
    x = torch.linspace(-5, 5, 101)
    assert torch.equal(target(x, 2.0, "left"), target(-x, 2.0, "right"))


def test_data_is_deterministic_and_paired_across_targets():
    a = make_data(torch.Generator().manual_seed(1), 3, "cpu", 1.0, "right")
    b = make_data(torch.Generator().manual_seed(1), 3, "cpu", 2.0, "left")
    # same inputs and same noise draws, only the target differs
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    assert torch.allclose(a[1] - target(a[0], 1.0, "right"), b[1] - target(b[0], 2.0, "left"), atol=1e-6)


def test_hard_mask_halves():
    Xt = make_data(torch.Generator().manual_seed(0), 1, "cpu")[4]
    assert int(hard_mask(Xt, "right").sum()) + int(hard_mask(Xt, "left").sum()) == N_TEST


def test_weights_shared_across_methods():
    a = draw_weights(torch.Generator().manual_seed(3), 2, 8, False)
    b = draw_weights(torch.Generator().manual_seed(3), 2, 8, True)
    for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
        assert torch.equal(a[k], b[k])
    assert b["Wo"].shape[1] == 9 and a["Wo"].shape[1] == 8


def _net(configs, blocks=1, **kw):
    w = draw_weights(torch.Generator().manual_seed(7), 2, 8, configs[0]["method"] == "grad_mag")
    return StackedNet(configs, w, blocks, 2, 2, "cpu", **kw)


def test_stacked_configs_equal_separate_networks():
    cfgs = [dict(method="addition", init=c, init2=s, c_fixed=None) for c, s in ((0.1, 0.25), (1.0, 0.01))]
    x = torch.linspace(-12, 12, 50).view(1, -1, 1)
    both = _net(cfgs)(x.repeat(4, 1, 1))
    one = torch.cat([_net([c])(x.repeat(2, 1, 1)) for c in cfgs], 0)
    assert torch.allclose(both, one, atol=1e-6)


def test_every_method_runs_and_is_finite():
    x = torch.linspace(-12, 12, 20).view(1, -1, 1).repeat(2, 1, 1)
    for (m, i, i2, cf) in method_configs(8, "F"):
        y = _net([dict(method=m, init=i, init2=i2, c_fixed=cf)])(x)
        assert y.shape == (2, 20, 1) and torch.isfinite(y).all()


def test_shuffle_keeps_norms_and_is_a_permutation():
    g = torch.randn(3, 40, 5)
    g[:, ::4] = 0
    gn = g / (g.norm(dim=-1, keepdim=True) + 1e-7)
    s = shuffle_nonzero(gn, torch.Generator().manual_seed(0))
    assert torch.allclose(s.norm(dim=-1), gn.norm(dim=-1))
    for i in range(3):
        rows = lambda t: sorted(map(tuple, t[i].round(decimals=5).tolist()))
        assert rows(s) == rows(gn)


def test_normalised_shortcut_has_unit_norm():
    net = _net([dict(method="standard_fixed", init=1.0, init2=None, c_fixed=None)], shortcut_norm="unit")
    q = net._source(torch.randn(2, 30, 8) * 50)
    assert torch.allclose(q.norm(dim=-1), torch.ones(2, 30), atol=1e-5)


def test_registered_experiments_and_grouping():
    units = EXPERIMENTS["lr_budget"]()
    assert len(units) == 3 * 2 * 2 * 38 - 3 * 2  # regular is trained under F only
    groups = group_units(units, max_nets=720, max_cfg=4)
    assert sum(len(g) for g in groups) == len(units)
    for g in groups:
        assert len({(u["method"], u["variant"], u["width"], u["f"]) for u in g}) == 1
    assert make_unit("x", "F", 8, 1.0, "regular")["shortcut_norm"] == "none"
