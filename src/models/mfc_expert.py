"""
MFC (Multiple Feature Construction) expert for CONAN-SchNet Step 2.

Built on EvoGP's `Transformation` problem (API verified against the installed
package). Mechanism:

  1. Evolve a population of SINGLE-output trees (output_len=1). Transformation's
     fitness is |Pearson corr(tree_output, target)| per individual.
  2. Select the q features that are individually high-correlation AND mutually
     decorrelated -- this is exactly Transformation.new_feature's greedy step
     (re-implemented inline here so we KEEP the selected q-tree sub-forest for
     inference and interpretability).
  3. Fit a ridge combiner (post-hoc) mapping the q constructed features -> Delta.

Fallback: if a bin has too few samples (or GP errors out), fit RidgeCV directly
on the 128-d embedding instead.

Note on the design doc: this is post-hoc ridge, not in-loop ridge. EvoGP's
`evaluate` is hard-wired to correlation and `new_feature` depends on it, so an
in-loop joint ridge would require subclassing both and breaking the per-tree
fitness paradigm. Documented deviation; in-loop ridge is left as a future option.
"""

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import RidgeCV


def _greedy_decorrelate(corr: Tensor, q: int) -> Tensor:
    """corr: (M, M) abs-correlation, diagonal zeroed. Return bool keep-mask, |keep|=q.

    Mirrors evogp Transformation.new_feature.worthy_correlation: repeatedly drop
    the higher-index member of the most-correlated remaining pair until q remain.
    """
    keep = torch.ones(corr.shape[0], dtype=torch.bool, device=corr.device)
    corr = corr.clone()
    while int(keep.sum()) > q:
        flat = torch.argmax(corr)
        i, j = torch.unravel_index(flat, corr.shape)
        worst = max(int(i), int(j))
        keep[worst] = False
        corr[worst, :] = 0
        corr[:, worst] = 0
    return keep


def _features_from_forest(forest, X: Tensor, n_feat: int) -> Tensor:
    """Run a forest of `n_feat` single-output trees on X -> (n_samples, n_feat)."""
    out = forest.batch_forward(X)                  # (n_feat, n_samples[, 1])
    out = out.reshape(out.shape[0], -1)            # (n_feat, n_samples)
    return out.T                                   # (n_samples, n_feat)


def _abs_corr(forest, X: Tensor, y: Tensor) -> Tensor:
    """|Pearson corr| of every individual's output with y. Returns (pop,), NaN->0.

    EvoGP's DefaultSelection sorts fitness DESCENDING (keeps highest), i.e. the GP
    MAXIMIZES fitness. So |corr| is used directly as fitness; nan_to_num maps
    constant-output trees to 0 (worst) so they are culled instead of poisoning the
    descending sort with NaN (which caused the population to collapse).
    """
    out = forest.batch_forward(X)
    out = out.reshape(out.shape[0], -1)            # (pop, n_samples)
    od = out - out.mean(dim=1, keepdim=True)
    ld = y - y.mean()
    num = (od * ld).sum(dim=1)
    den = torch.sqrt((od ** 2).sum(dim=1) * (ld ** 2).sum())
    return torch.nan_to_num((num / (den + 1e-12)).abs(), nan=0.0, posinf=0.0, neginf=0.0)


class MFCExpert:
    def __init__(self, cfg: dict, device: str = "cuda:0"):
        self.cfg = cfg
        self.device = device
        self.mode: Optional[str] = None            # 'gp' | 'ridge'
        self.forest = None                         # selected q-tree sub-forest (gp mode)
        self.n_feat: int = 0
        self.ridge: Optional[RidgeCV] = None
        self.input_len: int = 0

    # ------------------------------------------------------------------
    def fit(self, X: Tensor, delta: Tensor, gp_seed: int = 0,
            min_samples: int = 100) -> "MFCExpert":
        """X: (n, D) standardized embeddings. delta: (n,) residual target."""
        X = X.to(self.device).float()
        delta = delta.to(self.device).float().reshape(-1)
        self.input_len = int(X.shape[1])
        n = int(X.shape[0])

        if n < min_samples:
            print(f"    [expert] only {n} samples (<{min_samples}) -> ridge fallback")
            return self._fit_ridge(X, delta)

        try:
            return self._fit_gp(X, delta, gp_seed)
        except Exception as e:  # noqa: BLE001 - GP/CUDA failures should not abort the run
            print(f"    [expert] GP failed ({e!r}) -> ridge fallback")
            return self._fit_ridge(X, delta)

    def _fit_ridge(self, X: Tensor, delta: Tensor) -> "MFCExpert":
        self.mode = 'ridge'
        Xn = X.cpu().numpy()
        self.ridge = RidgeCV(alphas=self.cfg["ridge_alphas"]).fit(Xn, delta.cpu().numpy())
        return self

    def _fit_gp(self, X: Tensor, delta: Tensor, gp_seed: int) -> "MFCExpert":
        from evogp.tree import Forest, GenerateDescriptor
        from evogp.algorithm import (GeneticProgramming, DefaultCrossover,
                                      DefaultMutation, DefaultSelection)
        from evogp.problem import Transformation
        from evogp.pipeline import StandardPipeline

        torch.manual_seed(gp_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gp_seed)

        c = self.cfg
        desc = GenerateDescriptor(
            max_tree_len=c["max_tree_len"],        # 128 in evogp's sr_test
            input_len=self.input_len,
            output_len=1,                          # single-feature trees
            const_prob=c.get("const_prob", 0.5),
            out_prob=c.get("out_prob", 0.5),
            layer_leaf_prob=c.get("layer_leaf_prob", 0.2),
            max_layer_cnt=c["max_layer_cnt"],      # REQUIRED by evogp (gen depth)
            const_range=tuple(c["const_range"]),
            sample_cnt=c["sample_cnt"],            # REQUIRED with const_range
            using_funcs=c["using_funcs"],          # symbols, e.g. ['+','-','*','/']
        )
        forest = Forest.random_generate(pop_size=c["pop_size"], descriptor=desc)
        algo = GeneticProgramming(
            initial_forest=forest,
            crossover=DefaultCrossover(),
            # mutation uses a SHALLOWER descriptor (less destructive) -- as in sr_test
            mutation=DefaultMutation(
                mutation_rate=c["mutation_rate"],
                descriptor=desc.update(max_layer_cnt=c["mutation_max_layer_cnt"]),
            ),
            selection=DefaultSelection(survival_rate=c["survival_rate"],
                                       elite_rate=c["elite_rate"]),
            enable_pareto_front=False,
        )
        # EvoGP MAXIMIZES fitness (DefaultSelection keeps highest). Maximize |corr|
        # directly; nan_to_num so constant-output trees get fitness 0 (worst) and are
        # culled instead of poisoning the descending sort with NaN.
        class _CorrMax(Transformation):
            def evaluate(self, forest):
                return _abs_corr(forest, self.datapoints, self.labels)

        problem = _CorrMax(datapoints=X, labels=delta)
        StandardPipeline(algorithm=algo, problem=problem,
                         generation_limit=c["generation_limit"],
                         is_show_details=False).run()

        # NOTE: confirm `algo.forest` is the evolved population handle on first run.
        final_forest = getattr(algo, "forest", forest)

        corr = _abs_corr(final_forest, X, delta)                 # (pop,) |corr|, want HIGH
        n_best = min(c["n_best"], int(corr.shape[0]))
        best_idx = corr.argsort(descending=True)[:n_best]
        best_forest = final_forest[best_idx]

        F = best_forest.batch_forward(X).reshape(n_best, -1)     # (n_best, n)
        corr = torch.abs(torch.corrcoef(F))
        corr = torch.nan_to_num(corr, nan=0.0)
        corr.fill_diagonal_(0)

        q = min(c["q"], n_best)
        keep = _greedy_decorrelate(corr, q)
        self.forest = best_forest[keep]
        self.n_feat = int(keep.sum())
        self.mode = 'gp'

        feats = _features_from_forest(self.forest, X, self.n_feat).cpu().numpy()
        self.ridge = RidgeCV(alphas=c["ridge_alphas"]).fit(feats, delta.cpu().numpy())
        print(f"    [expert] GP ok: q={self.n_feat} features, "
              f"alpha={self.ridge.alpha_:.3g}")
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: Tensor) -> np.ndarray:
        """X: (m, D) standardized embeddings -> (m,) predicted Delta."""
        X = X.to(self.device).float()
        if self.mode == 'gp':
            feats = _features_from_forest(self.forest, X, self.n_feat).cpu().numpy()
            return self.ridge.predict(feats)
        return self.ridge.predict(X.cpu().numpy())

    # ------------------------------------------------------------------
    def export_expressions(self, symbol_names: Optional[List[str]] = None,
                           png_dir: Optional[str] = None) -> List[str]:
        """Return sympy strings for the q constructed features (gp mode only).

        Optionally also write a PNG per tree (needs graphviz; failures ignored).
        """
        if self.mode != 'gp' or self.forest is None:
            return []
        exprs = []
        # Forest is iterable over its trees in evogp.
        for k, tree in enumerate(self.forest):
            try:
                exprs.append(str(tree.to_sympy_expr(symbol_names=symbol_names)))
            except Exception as e:  # noqa: BLE001
                exprs.append(f"<sympy export failed: {e!r}>")
            if png_dir is not None:
                try:
                    import os
                    os.makedirs(png_dir, exist_ok=True)
                    tree.to_png(os.path.join(png_dir, f"cf_{k}.png"))
                except Exception:
                    pass
        return exprs

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {"mode": self.mode, "forest": self.forest, "n_feat": self.n_feat,
                "ridge": self.ridge, "input_len": self.input_len, "cfg": self.cfg}

    @classmethod
    def from_state(cls, sd: dict, device: str = "cuda:0") -> "MFCExpert":
        obj = cls(sd["cfg"], device=device)
        obj.mode = sd["mode"]
        obj.forest = sd["forest"]
        obj.n_feat = sd["n_feat"]
        obj.ridge = sd["ridge"]
        obj.input_len = sd["input_len"]
        return obj