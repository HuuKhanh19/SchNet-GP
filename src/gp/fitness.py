"""Protected operators + metrics cho GP head.

Operators được vectorize bằng numpy (eval cây chạy trên cả vector phân tử cùng lúc).
Tất cả "an toàn": không raise, không trả NaN/inf (clamp về giá trị hữu hạn) để cây
xấu chỉ bị fitness kém chứ không làm hỏng cả run.
"""

import numpy as np

_CLAMP = 1e6   # chặn biên độ output trung gian để tránh overflow lan truyền


def _finite(x):
    """Đưa NaN/inf về số hữu hạn, clamp biên độ."""
    x = np.nan_to_num(x, nan=0.0, posinf=_CLAMP, neginf=-_CLAMP)
    return np.clip(x, -_CLAMP, _CLAMP)


# --- Binary ---
def padd(a, b):
    return _finite(np.add(a, b))


def psub(a, b):
    return _finite(np.subtract(a, b))


def pmul(a, b):
    return _finite(np.multiply(a, b))


def pdiv(a, b):
    """Protected division: |b|<eps -> trả 1.0."""
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    safe = np.where(np.abs(b) < 1e-9, 1.0, b)
    out = np.where(np.abs(b) < 1e-9, 1.0, np.divide(a, safe))
    return _finite(out)


# --- Unary ---
def pabs(a):
    return _finite(np.abs(a))


def psquare(a):
    return _finite(np.square(a))


def psqrt(a):
    """Protected sqrt: sqrt(|a|)."""
    return _finite(np.sqrt(np.abs(a)))


def plog(a):
    """Protected log: log(|a| + eps)."""
    return _finite(np.log(np.abs(a) + 1e-9))


def ptanh(a):
    return _finite(np.tanh(a))


# Registry: tên trong config -> (callable, arity)
OPERATORS = {
    "add": (padd, 2),
    "sub": (psub, 2),
    "mul": (pmul, 2),
    "pdiv": (pdiv, 2),
    "abs": (pabs, 1),
    "square": (psquare, 1),
    "sqrt": (psqrt, 1),
    "log": (plog, 1),
    "tanh": (ptanh, 1),
}


# =============================================================================
# Metrics
# =============================================================================

def rmse(pred, target) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred, target) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(pred - target)))


def r2(pred, target) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def fitness_value(
    pred_std: np.ndarray,
    target_std: np.ndarray,
    genotype_nodes: int,
    parsimony_coef: float,
) -> float:
    """Fitness MINIMIZE = RMSE (target standardized) + parsimony*tổng_node.

    pred không hữu hạn -> phạt lớn (cây hỏng vẫn bị loại mà không crash run).
    """
    pred = np.asarray(pred_std, dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        return 1e6
    base = rmse(pred, target_std)
    return base + parsimony_coef * float(genotype_nodes)
