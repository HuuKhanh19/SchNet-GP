#!/usr/bin/env python
"""Extract EvoGP's WORKING GP recipe (selection/mutation/crossover/pop/gens).

Our GP degrades the best individual in both fitness directions -> operator/elitism
config is wrong. The docs say `python -m evogp.sr_test` converges, so copy that
exact recipe. This dumps the bundled example/test source + operator defaults.

    python scripts/probe_sr_recipe.py

Copy the full output back.
"""
import os
import glob
import inspect
import importlib
import traceback

import evogp

pkg = os.path.dirname(evogp.__file__)
print("evogp package dir:", pkg)

# 1) the bundled SR example/test the docs reference
print("\n=== try to import & dump known example/test modules ===")
for modname in ["evogp.sr_test", "evogp.test", "evogp.tests", "evogp.example",
                "evogp.examples", "evogp.demo"]:
    try:
        m = importlib.import_module(modname)
        print(f"\n----- source of {modname} ({getattr(m,'__file__','?')}) -----")
        print(inspect.getsource(m))
    except Exception as e:
        print(f"(skip {modname}: {type(e).__name__})")

# 2) scan the installed tree (and its parent) for a runnable pipeline recipe
print("\n=== scan .py files that build a StandardPipeline ===")
seen = 0
for root in {pkg, os.path.dirname(pkg)}:
    for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if "StandardPipeline(" in txt and ("GeneticProgramming(" in txt or "random_generate" in txt):
            seen += 1
            print(f"\n----- {f} -----")
            print(txt)
            if seen >= 4:
                break
    if seen >= 4:
        break
if seen == 0:
    print("  (no example with StandardPipeline found inside the installed package;")
    print("   look in the source repo's example/ dir, e.g. example/custom_sr.py)")

# 3) operator signatures + any defaults
print("\n=== operator signatures ===")
try:
    from evogp.algorithm import (GeneticProgramming, DefaultSelection,
                                 DefaultMutation, DefaultCrossover,
                                 TournamentSelection)
    from evogp.tree import Forest
    for o in [Forest.random_generate, GeneticProgramming.__init__,
              DefaultSelection.__init__, DefaultMutation.__init__,
              DefaultCrossover.__init__, TournamentSelection.__init__]:
        print(f"  {o.__qualname__}{inspect.signature(o)}")
except Exception:
    traceback.print_exc()

# 4) does DefaultSelection actually preserve elites unchanged? dump its source
print("\n=== DefaultSelection source (how elites are handled) ===")
try:
    import evogp.algorithm.selection.default as ds
    print(inspect.getsource(ds))
except Exception:
    try:
        from evogp.algorithm import DefaultSelection
        print(inspect.getsource(DefaultSelection))
    except Exception:
        traceback.print_exc()