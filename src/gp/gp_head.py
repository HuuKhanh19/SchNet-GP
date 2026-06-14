"""Exp 2 — Multi-tree GP head (DEAP) trên embedding 128-dim ĐÓNG BĂNG.

Một individual = q cây độc lập. Cây j chỉ đọc khối 16 dim thứ j của embedding
(partition random disjoint, cố định chung mọi individual) + ephemeral constants.
Mỗi cây xuất 1 scalar -> Φ (N, q). Merge bằng RIDGE closed-form (cho_solve) NGAY
TRONG fitness:
    w = (ΦᵀΦ + αI)⁻¹ Φᵀ t_center ,  pred = Φw + b
    fitness = RMSE(t, pred) + λ·(tổng node q cây)         (thấp = tốt)

Model selection theo VAL RMSE (không phải train) -> chống overfit của head.
α evolution cố định (vd 1.0); α cuối tune trên val grid logspace(-2,4) -> nếu feature
là noise, val tự chọn shrinkage mạnh -> w→0 -> pred→intercept≈0 -> graceful floor.

Vectorize numpy toàn bộ molecule (không loop python theo phân tử). Quyết định backend:
EvoGP KHÔNG cho ridge-in-the-loop per-individual nên dùng DEAP (xem exp-series-residual).
"""

from __future__ import annotations

import copy
import operator
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from deap import base, creator, gp, tools
from scipy.linalg import cho_factor, cho_solve


# =============================================================================
# Protected ops (numpy-vectorized)
# =============================================================================

def p_add(a, b): return np.add(a, b)
def p_sub(a, b): return np.subtract(a, b)
def p_mul(a, b): return np.multiply(a, b)


def p_div(a, b):
    b = np.asarray(b, dtype=np.float64)
    safe = np.abs(b) > 1e-6
    return np.where(safe, np.divide(a, np.where(safe, b, 1.0)), 1.0)


def p_log(a): return np.log(np.abs(a) + 1e-6)
def p_sqrt(a): return np.sqrt(np.abs(a))
def p_sin(a): return np.sin(a)
def p_cos(a): return np.cos(a)
def p_tanh(a): return np.tanh(a)

# Non-diff ops (Nhánh B — justify eggroll; backprop chết với những op này).
def p_gt(a, b): return (np.asarray(a) > np.asarray(b)).astype(np.float64)
def p_ifte(c, a, b): return np.where(np.asarray(c) > 0.0, a, b)
def p_hmin(a, b): return np.minimum(a, b)
def p_hmax(a, b): return np.maximum(a, b)
def p_step(a): return (np.asarray(a) > 0.0).astype(np.float64)


# name -> (callable numpy, arity).
_OPS = {
    "add": (p_add, 2), "sub": (p_sub, 2), "mul": (p_mul, 2), "pdiv": (p_div, 2),
    "sin": (p_sin, 1), "cos": (p_cos, 1), "plog": (p_log, 1),
    "psqrt": (p_sqrt, 1), "tanh": (p_tanh, 1),
    # non-diff:
    "gt": (p_gt, 2), "ifte": (p_ifte, 3), "hmin": (p_hmin, 2),
    "hmax": (p_hmax, 2), "step": (p_step, 1),
}
FUNCSET_DIFF = ["add", "sub", "mul", "pdiv", "sin", "cos", "plog", "psqrt", "tanh"]
FUNCSET_NONDIFF = FUNCSET_DIFF + ["gt", "ifte", "hmin", "hmax", "step"]
FUNCSET = FUNCSET_DIFF  # mặc định (Exp 2 + Nhánh A)


def funcset_list(name: str):
    return {"diff": FUNCSET_DIFF, "nondiff": FUNCSET_NONDIFF}[name]


def _erc():
    return random.gauss(0.0, 1.0) if random.random() < 0.5 else random.uniform(-2.0, 2.0)


# =============================================================================
# Config
# =============================================================================

@dataclass
class GPConfig:
    q: int = 8                       # số cây / individual (sweep {4,8,16})
    emb_dim: int = 128
    max_depth: int = 5               # giữ cây nông (4-6) -> low-capacity control
    init_min: int = 1
    init_max: int = 3
    pop_size: int = 1000
    generations: int = 100
    es_patience: int = 20            # early stop trên val
    cxpb: float = 0.9
    mutpb: float = 0.1
    tourn_size: int = 3
    parsimony: float = 1e-3          # λ phạt số node
    ridge_alpha: float = 1.0         # α cố định trong evolution
    seed: int = 0
    funcset: str = "diff"            # 'diff' (Exp2/Nhánh A) | 'nondiff' (Nhánh B)

    @property
    def block_dim(self) -> int:
        assert self.emb_dim % self.q == 0, "emb_dim phải chia hết cho q"
        return self.emb_dim // self.q


# =============================================================================
# Partition + primitive sets + compile cache
# =============================================================================

def make_partition(cfg: GPConfig) -> List[np.ndarray]:
    """Shuffle 128 chỉ số dim (seed cố định) rồi chia q khối block_dim. Cố định mọi individual."""
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(cfg.emb_dim)
    return [perm[j * cfg.block_dim:(j + 1) * cfg.block_dim] for j in range(cfg.q)]


def make_psets(cfg: GPConfig) -> List[gp.PrimitiveSet]:
    """q pset, cây j có arity = block_dim (đọc 16 dim khối j) + ERC riêng."""
    psets = []
    fset = funcset_list(cfg.funcset)
    for j in range(cfg.q):
        ps = gp.PrimitiveSet(f"B{j}", cfg.block_dim)
        ps.renameArguments(**{f"ARG{c}": f"x{c}" for c in range(cfg.block_dim)})
        for nm in fset:
            fn, ar = _OPS[nm]
            ps.addPrimitive(fn, ar, name=nm)
        ps.addEphemeralConstant(f"erc_{j}", _erc)
        psets.append(ps)
    return psets


class Compiler:
    """Memo gp.compile theo (id(pset), str(tree))."""

    def __init__(self):
        self.cache: Dict[Tuple[int, str], Callable] = {}

    def __call__(self, tree, pset) -> Callable:
        key = (id(pset), str(tree))
        f = self.cache.get(key)
        if f is None:
            f = gp.compile(tree, pset)
            self.cache[key] = f
        return f


def _vec(out, n: int) -> np.ndarray:
    a = np.asarray(out, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(n, float(a))
    return a


# =============================================================================
# Forward + ridge
# =============================================================================

def assemble_phi(ind, emb: np.ndarray, partition, psets, compiler) -> np.ndarray:
    """Φ (N, q): cây j eval trên 16 cột khối j của emb. nan/inf -> 0."""
    n, q = emb.shape[0], len(ind)
    phi = np.empty((n, q), dtype=np.float64)
    for j in range(q):
        cols = [emb[:, partition[j][c]] for c in range(len(partition[j]))]
        f = compiler(ind[j], psets[j])
        phi[:, j] = _vec(f(*cols), n)
    return np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)


def fit_ridge(phi: np.ndarray, t: np.ndarray, alpha: float):
    """Ridge closed-form, intercept = mean(t) (không penalize). Trả (w, b)."""
    b = float(np.mean(t))
    tc = t - b
    p = phi.shape[1]
    A = phi.T @ phi + alpha * np.eye(p)
    try:
        w = cho_solve(cho_factor(A, lower=True), phi.T @ tc)
    except Exception:
        w = np.linalg.lstsq(A, phi.T @ tc, rcond=None)[0]
    return w, b


def ridge_predict(phi, w, b):
    return phi @ w + b


def rmse(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# =============================================================================
# DEAP toolbox (multi-tree individual)
# =============================================================================

def _ensure_creator():
    if not hasattr(creator, "FitnessMinGP"):
        creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualGP"):
        creator.create("IndividualGP", list, fitness=creator.FitnessMinGP)


def build_toolbox(cfg: GPConfig, psets):
    _ensure_creator()
    tb = base.Toolbox()

    def make_tree(j):
        expr = gp.genHalfAndHalf(psets[j], min_=cfg.init_min, max_=cfg.init_max)
        return gp.PrimitiveTree(expr)

    def make_ind():
        return creator.IndividualGP([make_tree(j) for j in range(cfg.q)])

    tb.register("individual", make_ind)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("select", tools.selTournament, tournsize=cfg.tourn_size)

    def mate(a, b):
        """Lai 1 cây ở vị trí ngẫu nhiên (cùng pset)."""
        j = random.randrange(cfg.q)
        a[j], b[j] = gp.cxOnePoint(a[j], b[j])
        return a, b

    def mutate(ind):
        j = random.randrange(cfg.q)
        expr = lambda pset, type_=None: gp.genFull(pset, min_=0, max_=2)
        ind[j], = gp.mutUniform(ind[j], expr=expr, pset=psets[j])
        return (ind,)

    tb.register("mate", mate)
    tb.register("mutate", mutate)
    return tb


def _total_nodes(ind) -> int:
    return sum(len(t) for t in ind)


def _enforce_depth(ind, cfg: GPConfig):
    """Bỏ qua (giữ cây cũ) nếu vượt max_depth — dùng sau cx/mut."""
    return all(t.height <= cfg.max_depth for t in ind)


# =============================================================================
# Vòng evolution chính
# =============================================================================

def run_gp_head(emb_tr, t_tr, emb_va, t_va, emb_te, t_te, cfg: GPConfig,
                seed_individuals=None, verbose: bool = True) -> dict:
    """Tiến hóa q-tree GP head; chọn theo VAL RMSE (đơn vị target chuẩn hóa).

    target t_* đã standardize (zero-mean/unit-std bằng stats train) — denorm ở runner.
    Trả best individual + ridge (w,b) + dự đoán mọi split + lịch sử val.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    partition = make_partition(cfg)
    psets = make_psets(cfg)
    compiler = Compiler()
    tb = build_toolbox(cfg, psets)

    def evaluate(ind):
        """Fit ridge trên train; gắn ridge + val_rmse; trả fitness train + parsimony."""
        phi_tr = assemble_phi(ind, emb_tr, partition, psets, compiler)
        w, b = fit_ridge(phi_tr, t_tr, cfg.ridge_alpha)
        tr_rmse = rmse(t_tr, ridge_predict(phi_tr, w, b))
        phi_va = assemble_phi(ind, emb_va, partition, psets, compiler)
        ind.ridge = (w, b)
        ind.val_rmse = rmse(t_va, ridge_predict(phi_va, w, b))
        return (tr_rmse + cfg.parsimony * _total_nodes(ind),)

    pop = tb.population(n=cfg.pop_size)
    # Warm-start (slow-step Exp 3): gieo best individual cũ vào tối đa nửa quần thể.
    if seed_individuals:
        k = min(len(seed_individuals), cfg.pop_size // 2)
        for i in range(k):
            pop[i] = copy.deepcopy(seed_individuals[i % len(seed_individuals)])
    for ind in pop:
        ind.fitness.values = evaluate(ind)

    best = min(pop, key=lambda i: i.val_rmse)
    best = copy.deepcopy(best)
    best_val = best.val_rmse
    no_improve = 0
    history = [best_val]

    for gen in range(1, cfg.generations + 1):
        offspring = [copy.deepcopy(i) for i in tb.select(pop, len(pop))]
        for i in range(1, len(offspring), 2):
            if random.random() < cfg.cxpb:
                a, b = offspring[i - 1], offspring[i]
                ca, cb = copy.deepcopy(a), copy.deepcopy(b)
                tb.mate(a, b)
                if not (_enforce_depth(a, cfg) and _enforce_depth(b, cfg)):
                    offspring[i - 1], offspring[i] = ca, cb  # rollback nếu quá sâu
                else:
                    del a.fitness.values, b.fitness.values
        for i in range(len(offspring)):
            if random.random() < cfg.mutpb:
                ind = offspring[i]
                c = copy.deepcopy(ind)
                tb.mutate(ind)
                if not _enforce_depth(ind, cfg):
                    offspring[i] = c
                elif hasattr(ind.fitness, "values"):
                    if ind.fitness.valid:
                        del ind.fitness.values

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)

        # (μ+λ) elitism theo train-fitness để giữ áp lực; chọn best theo VAL.
        pop = tools.selBest(pop + offspring, cfg.pop_size)
        gen_best = min(pop, key=lambda i: i.val_rmse)
        if gen_best.val_rmse < best_val - 1e-9:
            best_val = gen_best.val_rmse
            best = copy.deepcopy(gen_best)
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_val)
        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"    gen {gen:3d} | best val_rmse(std)={best_val:.4f} | "
                  f"nodes={_total_nodes(best)} | no_improve={no_improve}")
        if no_improve >= cfg.es_patience:
            if verbose:
                print(f"    early stop @ gen {gen} (best val_rmse(std)={best_val:.4f})")
            break

    # α cuối tune trên val (grid lớn -> graceful shrinkage nếu feature là noise).
    phi_tr = assemble_phi(best, emb_tr, partition, psets, compiler)
    phi_va = assemble_phi(best, emb_va, partition, psets, compiler)
    phi_te = assemble_phi(best, emb_te, partition, psets, compiler)
    best_alpha, best_a_val = cfg.ridge_alpha, float("inf")
    for alpha in np.logspace(-2, 4, 13):
        w, b = fit_ridge(phi_tr, t_tr, alpha)
        v = rmse(t_va, ridge_predict(phi_va, w, b))
        if v < best_a_val:
            best_a_val, best_alpha = v, float(alpha)
    w, b = fit_ridge(phi_tr, t_tr, best_alpha)

    return {
        "best_ind": best,
        "partition": partition,
        "ridge_w": w, "ridge_b": b, "alpha": best_alpha,
        "val_rmse_std": best_a_val,
        "pred_tr_std": ridge_predict(phi_tr, w, b),
        "pred_va_std": ridge_predict(phi_va, w, b),
        "pred_te_std": ridge_predict(phi_te, w, b),
        "history": history,
        "trees_str": [str(t) for t in best],
    }
