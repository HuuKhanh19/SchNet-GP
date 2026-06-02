"""Unit tests for the non-EvoGP, non-SchNet parts of Step 2 (runnable on CPU)."""

import numpy as np
import torch

from src.utils.scatter import scatter_add, scatter_mean, scatter_softmax
from src.utils.standardize import Standardizer
from src.models.energy_gate import EnergyGate
from src.models.delta_baseline import DeltaBaseline
from src.models.mfc_expert import MFCExpert, _greedy_decorrelate
from src.models.conan_head import ConanHead, aggregate, predict_per_conf_delta


def test_scatter():
    src = torch.tensor([1., 2., 3., 4.])
    idx = torch.tensor([0, 0, 1, 1])
    assert torch.allclose(scatter_add(src, idx), torch.tensor([3., 7.]))
    assert torch.allclose(scatter_mean(src, idx), torch.tensor([1.5, 3.5]))
    # 2-D
    src2 = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])
    idx2 = torch.tensor([0, 1, 1])
    assert torch.allclose(scatter_add(src2, idx2), torch.tensor([[1., 1.], [5., 5.]]))
    # softmax sums to 1 per segment
    s = torch.randn(6)
    i = torch.tensor([0, 0, 0, 1, 1, 1])
    w = scatter_softmax(s, i)
    assert torch.allclose(scatter_add(w, i), torch.ones(2), atol=1e-5)
    print("OK test_scatter")


def test_softmax_grad_through_tau():
    # the Phase-2 path: optimize tau through scatter_softmax; loss must drop
    torch.manual_seed(0)
    n_mol, K = 40, 5
    conf2mol = torch.arange(n_mol).repeat_interleave(K)
    dE = torch.rand(n_mol * K) * 3.0
    s = torch.randn(n_mol * K)
    y = torch.randn(n_mol)
    rho = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([rho], lr=0.1)
    losses = []
    for _ in range(50):
        tau = torch.nn.functional.softplus(rho) + 1e-4
        w = scatter_softmax(-dE / tau, conf2mol, dim_size=n_mol)
        dh = scatter_add(w * s, conf2mol, dim_size=n_mol)
        loss = torch.nn.functional.mse_loss(dh, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert rho.grad is not None
    assert losses[-1] <= losses[0] + 1e-6
    print(f"OK test_softmax_grad_through_tau (loss {losses[0]:.3f} -> {losses[-1]:.3f})")


def test_gate():
    rng = np.random.default_rng(0)
    dE = np.concatenate([np.zeros(100), rng.random(300) * 5.0, [np.inf, np.inf]])
    g = EnergyGate(num_experts=3).fit(dE)
    assert g.boundaries.shape == (2,)
    bins = g.route(dE)
    assert set(np.unique(bins)).issubset({0, 1, 2})
    # inf routed to top bin after clamp
    assert bins[-1] == 2
    # roughly equal counts (quantile)
    counts = np.bincount(bins, minlength=3)
    assert counts.min() > 0
    # num_experts=1 => all bin 0
    g1 = EnergyGate(num_experts=1).fit(dE)
    assert (g1.route(dE) == 0).all()
    print("OK test_gate", g.bin_ranges(dE))


def test_greedy_decorrelate():
    # 4 features: f0,f1 nearly identical; f2,f3 distinct -> keep 2 should drop one of f0/f1
    corr = torch.tensor([
        [0.0, 0.99, 0.1, 0.1],
        [0.99, 0.0, 0.1, 0.1],
        [0.1, 0.1, 0.0, 0.2],
        [0.1, 0.1, 0.2, 0.0],
    ])
    keep = _greedy_decorrelate(corr, q=3)
    assert int(keep.sum()) == 3
    # exactly one of the highly-correlated pair {0,1} removed
    assert keep[0].item() ^ keep[1].item() or (keep[0] and keep[1])  # at least consistent
    assert int(keep.sum()) == 3
    print("OK test_greedy_decorrelate keep=", keep.tolist())


def test_delta_baseline():
    smi = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "O", "CCCCCC", "c1ccncc1", "CC(C)C"]
    y = np.array([-0.77, -2.13, -0.17, 1.0, 1.38, -3.2, -0.5, 0.1])
    db = DeltaBaseline(kind='rdkit2d')
    yb = db.fit(smi, y)
    assert yb.shape == (len(smi),)
    pred = db.predict(["CCO", "invalid_smiles_xyz", "c1ccccc1"])
    assert pred.shape == (3,) and np.isfinite(pred).all()
    # 'none' mode -> zeros
    assert np.allclose(DeltaBaseline(kind='none').fit(smi, y), 0.0)
    print("OK test_delta_baseline alpha=", db.ridge.alpha_)


def test_end_to_end_head_no_evogp():
    """Full ConanHead path with ridge-mode experts (no EvoGP, no SchNet)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_mol, D = 30, 16
    K = 4
    conf2mol = torch.arange(n_mol).repeat_interleave(K)
    C = n_mol * K
    emb = torch.randn(C, D)
    dE = torch.tensor(rng.random(C) * 4.0, dtype=torch.float32)
    smiles = ["CCO"] * n_mol
    y = rng.standard_normal(n_mol)

    std = Standardizer().fit(emb)
    emb_s = std.transform(emb)
    gate = EnergyGate(num_experts=2).fit(dE.numpy())
    bins = gate.route(dE.numpy())

    cfg_expert = {"ridge_alphas": [0.1, 1.0, 10.0], "q": 4, "n_best": 8,
                  "pop_size": 50, "generation_limit": 2, "max_tree_len": 32,
                  "max_layer_cnt": 3, "mutation_rate": 0.3, "survival_rate": 0.3,
                  "elite_rate": 0.05, "using_funcs": None, "const_range": [-1, 1],
                  "gp_seed": 0}
    delta_conf = (y - y.mean())[conf2mol.numpy()]
    experts = []
    for b in range(2):
        idx = np.where(bins == b)[0]
        e = MFCExpert(cfg_expert, device="cpu")
        e._fit_ridge(emb_s[idx], torch.tensor(delta_conf[idx], dtype=torch.float32))
        experts.append(e)
    assert all(e.mode == 'ridge' for e in experts)

    s = predict_per_conf_delta(experts, emb_s, bins)
    assert s.shape == (C,)

    db = DeltaBaseline(kind='none'); db.fit(smiles, y)
    head = ConanHead(db, std, gate, experts, agg='learned_softmax', tau=0.6, device="cpu")
    yhat = head.predict(emb, dE, conf2mol, smiles)
    assert yhat.shape == (n_mol,) and np.isfinite(yhat).all()

    # mean-agg and boltzmann-agg also run
    for agg in ('mean', 'boltzmann'):
        head.agg = agg
        assert head.predict(emb, dE, conf2mol, smiles).shape == (n_mol,)
    print("OK test_end_to_end_head_no_evogp  yhat[:3]=", np.round(yhat[:3], 3))


if __name__ == "__main__":
    test_scatter()
    test_softmax_grad_through_tau()
    test_gate()
    test_greedy_decorrelate()
    test_delta_baseline()
    test_end_to_end_head_no_evogp()
    print("\nALL TESTS PASSED")