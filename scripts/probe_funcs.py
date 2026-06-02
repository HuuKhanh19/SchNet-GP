#!/usr/bin/env python
"""Discover the valid `using_funcs` for evogp.GenerateDescriptor.

    python scripts/probe_funcs.py

Copy the full output back.
"""
import inspect
import traceback

import evogp.tree as T

print("=== evogp.tree members ===")
print([x for x in dir(T) if not x.startswith("_")])

# 1) NType enum -- almost certainly the canonical function identifiers
print("\n=== NType ===")
try:
    NType = T.NType
    members = list(NType)
    for m in members:
        print(f"  {m!r}  value={getattr(m, 'value', '?')}")
except Exception:
    traceback.print_exc()

# 2) how check_tree_length consumes using_funcs (mapping/validation)
print("\n=== descriptor.check_tree_length source ===")
try:
    import evogp.tree.descriptor as d
    print(inspect.getsource(d.check_tree_length))
except Exception:
    traceback.print_exc()

# 3) the using_funcs handling block in GenerateDescriptor.__init__
print("\n=== GenerateDescriptor.__init__ source (first 4000 chars) ===")
try:
    print(inspect.getsource(T.GenerateDescriptor.__init__)[:4000])
except Exception:
    traceback.print_exc()

# 4) brute-force which formats actually CONSTRUCT a descriptor
print("\n=== try candidate using_funcs formats ===")
candidates = {
    "list_symbols":      ["+", "-", "*", "/"],
    "list_sym_trans":    ["+", "-", "*", "/", "sin", "cos", "exp", "log"],
    "dict_symbols":      {"+": 1.0, "-": 1.0, "*": 1.0, "/": 1.0},
    "list_names":        ["add", "sub", "mul", "div"],
    "list_names_trans":  ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
}
# also try NType members directly if available
try:
    nt = list(T.NType)
    candidates["list_NType_all"] = nt
except Exception:
    pass

for label, funcs in candidates.items():
    try:
        T.GenerateDescriptor(max_tree_len=64, input_len=16, output_len=1,
                             max_layer_cnt=4, const_range=(-1, 1), using_funcs=funcs)
        print(f"  [WORKS] {label}: {funcs}")
    except Exception as e:
        print(f"  [fail ] {label}: {repr(e)[:90]}")