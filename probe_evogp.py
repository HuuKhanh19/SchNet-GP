#!/usr/bin/env python
"""
Probe the INSTALLED EvoGP API for CONAN-SchNet Step 2.

The design doc gives an API sketch but explicitly says "verify qua source".
This script discovers the truth from whatever is installed (not from docs),
so the Step-2 expert + fitness can be finalized. We need to nail down:

  - import paths + class names (Forest, GenerateDescriptor, Transformation,
    StandardPipeline, GeneticProgramming, Problem, ...)
  - Transformation: what it optimizes, and HOW to read out the q constructed
    features as a tensor (new_feature? transform? forward?)
  - multi-output trees via output_len
  - feasibility of a CUSTOM fitness / in-loop ridge (override Problem.evaluate?)
  - seeding + determinism (custom CUDA kernels on sm_120)
  - interpretability hooks (to_sympy_expr / to_png)

This script is standalone (only needs evogp + torch). Run on the TARGET
machine (sm_120 GPU, env conan_es), from anywhere:

    python probe_evogp.py

Everything is wrapped in try/except; ASCII-only (safe for PowerShell).
The signatures + source dump are the real payload -- the synthetic run at
the end is best-effort and MAY fail on guessed arg names; that's fine.
Copy the FULL output back.
"""

import sys
import inspect
import importlib
import traceback


def hr(t=""):
    print("\n" + "=" * 72)
    if t:
        print(t)
        print("=" * 72)


def show(name, obj, max_doc=700):
    try:
        sig = str(inspect.signature(obj))
    except Exception:
        sig = "(signature unavailable)"
    print(f"\n  >>> {name}{sig}")
    doc = inspect.getdoc(obj)
    if doc:
        head = "\n      ".join(doc.strip()[:max_doc].splitlines())
        print("      " + head)


# ----------------------------------------------------------------- import
hr("0. import evogp")
try:
    import evogp
    print("  evogp file   :", getattr(evogp, "__file__", "?"))
    print("  evogp version:", getattr(evogp, "__version__", "?"))
    print("  top-level    :", [x for x in dir(evogp) if not x.startswith("_")])
except Exception:
    traceback.print_exc()
    sys.exit("FATAL: cannot import evogp")

# -------------------------------------------------------------- submodules
TARGETS = {
    "evogp.tree":      ["Forest", "Tree", "GenerateDescriptor", "GenerateDiscriptor"],
    "evogp.algorithm": ["GeneticProgramming", "DefaultSelection",
                        "DefaultMutation", "DefaultCrossover"],
    "evogp.problem":   ["SymbolicRegression", "Classification",
                        "Transformation", "Problem"],
    "evogp.pipeline":  ["StandardPipeline"],
    "evogp.operator":  [],
}
METHODS_OF_INTEREST = [
    "new_feature", "transform", "forward", "evaluate", "fitness",
    "to_sympy_expr", "to_png", "predict", "run", "fit", "random_generate",
]

for mod, names in TARGETS.items():
    hr(f"submodule: {mod}")
    try:
        m = importlib.import_module(mod)
        print("  members:", [x for x in dir(m) if not x.startswith("_")])
        for n in names:
            obj = getattr(m, n, None)
            if obj is None:
                continue
            show(f"{mod}.{n}", obj)
            for meth in METHODS_OF_INTEREST:
                f = getattr(obj, meth, None)
                if callable(f):
                    try:
                        ms = str(inspect.signature(f))
                    except Exception:
                        ms = "(sig?)"
                    print(f"        .{meth}{ms}")
    except Exception:
        traceback.print_exc()

# ----------------------------------------- Transformation / Problem source
hr("Transformation source (custom-fitness + feature readout)")
try:
    from evogp.problem import Transformation
    print(inspect.getsource(Transformation)[:4500])
except Exception:
    traceback.print_exc()

hr("Problem.evaluate source (can we override for in-loop ridge?)")
try:
    from evogp.problem import Problem
    print(inspect.getsource(Problem)[:3500])
except Exception:
    traceback.print_exc()

# ----------------------------------------- minimal end-to-end (BEST EFFORT)
hr("minimal Transformation run on synthetic data -- BEST EFFORT")
print("  goal: confirm we can pull q constructed features out as a tensor,")
print("        and that output_len controls the number of features.")
print("  NOTE: arg names below are GUESSES from the doc; if this block fails,")
print("        read the signatures/source dumped above and tell me the real ones.")
try:
    import torch
    from evogp.tree import Forest, GenerateDescriptor
    from evogp.problem import Transformation
    from evogp.algorithm import (GeneticProgramming, DefaultSelection,
                                 DefaultMutation, DefaultCrossover)
    from evogp.pipeline import StandardPipeline

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    n, input_len, q = 256, 8, 4
    X = torch.randn(n, input_len, device=dev)
    w = torch.randn(input_len, device=dev)
    y = (X @ w + 0.1 * torch.randn(n, device=dev)).reshape(-1, 1)
    print(f"  X={tuple(X.shape)} y={tuple(y.shape)} input_len={input_len} q(output_len)={q} dev={dev}")

    desc = GenerateDescriptor(
        max_tree_len=64, input_len=input_len, output_len=q,
        # using_funcs=..., max_layer_cnt=..., const_samples=...
    )
    print("  GenerateDescriptor OK ->", type(desc))

    prob = Transformation(datapoints=X, labels=y)   # GUESS
    print("  Transformation OK ->", type(prob))

    algo = GeneticProgramming(
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(),
        selection=DefaultSelection(),
    )
    pipe = StandardPipeline(algorithm=algo, problem=prob, generation_limit=10)
    best = pipe.run()
    print("  pipeline ran. best ->", type(best))

    for meth in ["new_feature", "transform", "forward"]:
        f = getattr(best, meth, None)
        if callable(f):
            try:
                feats = f(X)
                print(f"  best.{meth}(X) -> shape {tuple(getattr(feats, 'shape', []))}  "
                      f"(want [{n}, {q}])")
                break
            except Exception as e:
                print(f"  best.{meth}(X) failed: {e!r}")
    for meth in ["to_sympy_expr", "to_png"]:
        if callable(getattr(best, meth, None)):
            print(f"  interpretability hook present: best.{meth}")
except Exception:
    traceback.print_exc()

# ----------------------------------------------------------- determinism
hr("determinism note")
print("  Run the synthetic block twice with the SAME seed and compare best")
print("  fitness. If results differ, multi-seed mean+/-std is mandatory and")
print("  per-seed variance must be reported. Look for a seed= arg on")
print("  GeneticProgramming / StandardPipeline / random_generate above.")

hr("DONE -- copy the full output back")