"""GP head cho SchNet-GP.

Pipeline: encoder SchNet (freeze, train 1-conf) -> extract conf embedding + descriptor
-> cache -> DEAP multi-tree GP head. Xem `src/gp/features.py` (PHASE 2) và
`src/gp/gp_head.py` (PHASE 3).
"""
