"""Eggroll: hard-count head trên SchNet warm-start (pure-encoder).

Pipeline (xem plan + spec §0-§12):
    encoder (SchNet frozen + LoRA) -> per-atom h -> hard-count head (Heaviside rules)
    -> delta ridge readout 2 tầng -> RMSE; head + adapter tối ưu bằng eggroll (ES).
"""
