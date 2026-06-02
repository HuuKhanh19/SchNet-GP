#!/usr/bin/env python
"""Final pre-run check: GP must IMPROVE |corr| with the corrected setup.

Truth (from DefaultSelection source): EvoGP MAXIMIZES fitness. So we maximize
|corr| directly, with NaN->0 so constant trees can't poison the descending sort.
Operators mirror evogp's working sr_test (mutation uses a shallower descriptor).
Target is exactly representable with + - * so |corr| can approach 1.0.

    python scripts/check_forest_handle.py
"""
import torch
from evogp.tree import Forest, GenerateDescriptor
from evogp.algorithm import (GeneticProgramming, DefaultCrossover,
                             DefaultMutation, DefaultSelection)
from evogp.problem import Transformation
from evogp.pipeline import StandardPipeline

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N, D, POP, GENS = 600, 6, 1000, 60


def abs_corr(forest, X, y):
    out = forest.batch_forward(X)
    out = out.reshape(out.shape[0], -1)              # (P, n)
    od = out - out.mean(dim=1, keepdim=True)
    ld = y - y.mean()
    num = (od * ld).sum(dim=1)
    den = torch.sqrt((od ** 2).sum(dim=1) * (ld ** 2).sum())
    return torch.nan_to_num((num / (den + 1e-12)).abs(), nan=0.0)


class CorrMax(Transformation):          # MAXIMIZE |corr| (framework keeps highest)
    def evaluate(self, forest):
        return abs_corr(forest, self.datapoints, self.labels)


torch.manual_seed(0)
X = torch.randn(N, D, device=DEV)
y = (X[:, 0] * X[:, 1] + X[:, 2] - 0.5 * X[:, 3]).to(DEV)   # representable with + - *

desc = GenerateDescriptor(max_tree_len=128, input_len=D, output_len=1,
                          const_prob=0.5, out_prob=0.5, layer_leaf_prob=0.2,
                          max_layer_cnt=5, const_range=(-3, 3),
                          sample_cnt=8, using_funcs=["+", "-", "*", "/"])
init = Forest.random_generate(pop_size=POP, descriptor=desc)
algo = GeneticProgramming(
    initial_forest=init,
    crossover=DefaultCrossover(),
    mutation=DefaultMutation(mutation_rate=0.2,
                             descriptor=desc.update(max_layer_cnt=3)),  # shallower (sr_test)
    selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
)
problem = CorrMax(datapoints=X, labels=y)

before = float(abs_corr(init, X, y).max())
print(f"Target = X0*X1 + X2 - 0.5*X3  (POP={POP}, GENS={GENS})")
print(f"initial best |corr| = {before:.4f}\n  ... running ...")
StandardPipeline(algorithm=algo, problem=problem,
                 generation_limit=GENS, is_show_details=False).run()
final = getattr(algo, "forest", init)
after = float(abs_corr(final, X, y).max())

print(f"\n  before |corr| = {before:.4f}")
print(f"  after  |corr| = {after:.4f}   ({'UP' if after > before else 'DOWN'})")
print("\nVerdict:")
if after > 0.9:
    print(f"  >> GP converges (|corr| -> {after:.3f}). Setup correct. RUN STEP 2.")
elif after > before + 0.05:
    print(f"  >> GP improves ({before:.3f} -> {after:.3f}) but not near 1.0; "
          f"consider more GENS. Still usable -- can RUN STEP 2.")
else:
    print(f"  >> Still not improving ({before:.3f} -> {after:.3f}). Paste output back.")

print("\n--- algo Forest attributes (best |corr|) ---")
for name in dir(algo):
    if name.startswith("_"):
        continue
    obj = getattr(algo, name, None)
    if isinstance(obj, Forest):
        try:
            print(f"  algo.{name:<14s} best|corr|={float(abs_corr(obj, X, y).max()):.4f}")
        except Exception as e:
            print(f"  algo.{name:<14s} (eval failed: {e!r})")

# show a best evolved expression (interpretability sanity)
try:
    best_tree = final[int(abs_corr(final, X, y).argmax())]
    print("\nbest evolved tree:", best_tree.to_sympy_expr())
except Exception as e:
    print("(to_sympy_expr failed:", repr(e), ")")