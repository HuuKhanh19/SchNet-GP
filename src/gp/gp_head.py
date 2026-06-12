"""PHASE 3 — DEAP multi-tree GP head (no tree-sharing).

Genotype = list các gp.PrimitiveTree, vai trò cố định theo vị trí:
    [K x q cây L1] + [K cây L2] + [1 cây L3]
với q = num_emb + num_desc3d.

  - L1 (bin i, slot j): nén feature -> 1 scalar v_j.
      * slot embedding j: đọc d chiều subspace[j] của conf_emb (subspace CHUNG mọi bin).
      * slot desc3d:      đọc toàn bộ 3D descriptor của bin i.
  - L2 (bin i): gom q scalar v_1..v_q -> s_i.
  - L3 (global): [s_0..s_{K-1}] + desc2d -> prediction.

Forward VECTORIZE qua numpy (không loop python theo phân tử). Routing: conformer đã
được sort theo energy tăng dần ở khâu extract, nên trục K chính là hạng energy -> bin i
= cột i. Fitness = RMSE trên target standardized (denormalize khi báo cáo).
"""

from __future__ import annotations

import operator
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from deap import base, creator, gp, tools


# =============================================================================
# Protected ops (numpy-vectorized) + ERC
# =============================================================================

def p_add(a, b): return np.add(a, b)
def p_sub(a, b): return np.subtract(a, b)
def p_mul(a, b): return np.multiply(a, b)


def p_div(a, b):
    b = np.asarray(b, dtype=np.float64)
    safe = np.abs(b) > 1e-6
    return np.where(safe, np.divide(a, np.where(safe, b, 1.0)), 1.0)


def p_sqrt(a): return np.sqrt(np.abs(a))
def p_log(a): return np.log(np.abs(a) + 1e-6)
def p_square(a): return np.square(a)
def p_abs(a): return np.abs(a)
def p_neg(a): return np.negative(a)
def p_tanh(a): return np.tanh(a)
def p_min(a, b): return np.minimum(a, b)
def p_max(a, b): return np.maximum(a, b)


# Function set theo lớp (tên -> (callable, arity)).
_OPS = {
    "add": (p_add, 2), "sub": (p_sub, 2), "mul": (p_mul, 2),
    "pdiv": (p_div, 2), "square": (p_square, 1), "psqrt": (p_sqrt, 1),
    "abs": (p_abs, 1), "neg": (p_neg, 1), "tanh": (p_tanh, 1),
    "plog": (p_log, 1), "min": (p_min, 2), "max": (p_max, 2),
}
FUNCSET_L1 = ["add", "sub", "mul", "pdiv", "square", "psqrt",
              "abs", "neg", "tanh", "plog", "min", "max"]
FUNCSET_L2 = ["add", "sub", "mul", "pdiv", "tanh"]
FUNCSET_L3 = ["add", "sub", "mul", "pdiv", "tanh", "square", "max", "min"]


def _erc():
    """ERC: 50% N(0,1), 50% uniform[-2,2]."""
    return random.gauss(0.0, 1.0) if random.random() < 0.5 else random.uniform(-2.0, 2.0)


_ERC_COUNTER = [0]


def _add_funcs(pset: gp.PrimitiveSet, names: List[str]) -> None:
    for nm in names:
        fn, ar = _OPS[nm]
        pset.addPrimitive(fn, ar, name=nm)
    # ERC tên duy nhất toàn cục (DEAP đăng ký class ephemeral theo tên).
    _ERC_COUNTER[0] += 1
    pset.addEphemeralConstant(f"erc_{_ERC_COUNTER[0]}", _erc)


# =============================================================================
# Config
# =============================================================================

@dataclass
class GPConfig:
    K: int = 8
    num_emb: int = 7
    num_desc3d: int = 2
    d: int = 16                 # số chiều mỗi subspace embedding
    num_2d: int = 8
    pop_size: int = 300
    generations: int = 100
    cxpb: float = 0.7
    mutpb: float = 0.2
    tourn_fitness_size: int = 5
    tourn_parsimony: float = 1.4
    seed_prob: float = 0.15     # tỉ lệ cá thể gieo prior ensemble-mean
    warmup: int = 0
    height_l1: int = 8
    height_l2: int = 6
    height_l3: int = 6
    init_l1: Tuple[int, int] = (2, 4)
    init_l23: Tuple[int, int] = (1, 3)
    seed: int = 0

    @property
    def q(self) -> int:
        return self.num_emb + self.num_desc3d


# Layer tag
L1, L2, L3 = 0, 1, 2


# =============================================================================
# Genotype layout
# =============================================================================

class Layout:
    """Ánh xạ vị trí cây <-> (layer, bin, slot) cho genotype K·(q+1)+1 cây."""

    def __init__(self, cfg: GPConfig):
        self.cfg = cfg
        self.Kq = cfg.K * cfg.q
        self.n_l2 = cfg.K
        self.total = self.Kq + cfg.K + 1
        self.idx_l3 = self.total - 1

    def l1(self, bin_i: int, slot_j: int) -> int:
        return bin_i * self.cfg.q + slot_j

    def l2(self, bin_i: int) -> int:
        return self.Kq + bin_i

    def layer_of(self, k: int) -> int:
        if k < self.Kq:
            return L1
        if k < self.Kq + self.cfg.K:
            return L2
        return L3

    def positions(self, layer: int) -> List[int]:
        if layer == L1:
            return list(range(self.Kq))
        if layer == L2:
            return list(range(self.Kq, self.Kq + self.cfg.K))
        return [self.idx_l3]


# =============================================================================
# Primitive sets
# =============================================================================

class PSets:
    """Tập hợp pset theo vai trò + ánh xạ vị trí -> pset."""

    def __init__(self, cfg: GPConfig, layout: Layout, subspaces: List[np.ndarray],
                 n3d: int):
        self.cfg = cfg
        self.layout = layout
        # pset L1 embedding theo slot j (arity = d). CHUNG cho mọi bin của slot j.
        self.emb = []
        for j in range(cfg.num_emb):
            ps = gp.PrimitiveSet(f"L1emb{j}", cfg.d)
            ps.renameArguments(**{f"ARG{c}": f"x{c}" for c in range(cfg.d)})
            _add_funcs(ps, FUNCSET_L1)
            self.emb.append(ps)
        # pset L1 desc3d (arity = n3d), chung cho num_desc3d slot.
        self.desc3d = gp.PrimitiveSet("L1desc3d", n3d)
        self.desc3d.renameArguments(**{f"ARG{c}": f"g{c}" for c in range(n3d)})
        _add_funcs(self.desc3d, FUNCSET_L1)
        # pset L2 (arity = q): biến trừu tượng v_0..v_{q-1}.
        self.l2 = gp.PrimitiveSet("L2", cfg.q)
        self.l2.renameArguments(**{f"ARG{c}": f"v{c}" for c in range(cfg.q)})
        _add_funcs(self.l2, FUNCSET_L2)
        # pset L3 (arity = K + num_2d): s_0..s_{K-1} + desc2d.
        self.l3 = gp.PrimitiveSet("L3", cfg.K + cfg.num_2d)
        names = {f"ARG{i}": f"s{i}" for i in range(cfg.K)}
        names.update({f"ARG{cfg.K + j}": f"m{j}" for j in range(cfg.num_2d)})
        self.l3.renameArguments(**names)
        _add_funcs(self.l3, FUNCSET_L3)

        # Ánh xạ vị trí -> pset
        self.by_pos: List[gp.PrimitiveSet] = [None] * layout.total
        for i in range(cfg.K):
            for j in range(cfg.q):
                self.by_pos[layout.l1(i, j)] = (
                    self.emb[j] if j < cfg.num_emb else self.desc3d
                )
            self.by_pos[layout.l2(i)] = self.l2
        self.by_pos[layout.idx_l3] = self.l3


# =============================================================================
# Compile cache + forward (vectorized fitness)
# =============================================================================

class Compiler:
    """gp.compile có memo theo (id(pset), str(tree)) -> tránh recompile cây không đổi."""

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


class Forward:
    """Tính prediction (standardized) cho 1 cá thể trên 1 split, vectorize numpy."""

    def __init__(self, cfg: GPConfig, layout: Layout, psets: PSets,
                 subspaces: List[np.ndarray], compiler: Compiler):
        self.cfg = cfg
        self.layout = layout
        self.psets = psets
        self.subspaces = subspaces
        self.compile = compiler

    def predict(self, ind, emb: np.ndarray, d3d: np.ndarray,
                d2d: np.ndarray) -> np.ndarray:
        cfg, lo = self.cfg, self.layout
        N = emb.shape[0]
        S = np.empty((N, cfg.K), dtype=np.float64)
        for i in range(cfg.K):
            V = np.empty((N, cfg.q), dtype=np.float64)
            for j in range(cfg.num_emb):
                sub = self.subspaces[j]
                cols = [emb[:, i, sub[c]] for c in range(cfg.d)]
                f = self.compile(ind[lo.l1(i, j)], self.psets.emb[j])
                V[:, j] = _vec(f(*cols), N)
            for jd in range(cfg.num_desc3d):
                j = cfg.num_emb + jd
                cols = [d3d[:, i, c] for c in range(d3d.shape[2])]
                f = self.compile(ind[lo.l1(i, j)], self.psets.desc3d)
                V[:, j] = _vec(f(*cols), N)
            f2 = self.compile(ind[lo.l2(i)], self.psets.l2)
            S[:, i] = _vec(f2(*[V[:, j] for j in range(cfg.q)]), N)

        cols3 = [S[:, i] for i in range(cfg.K)] + [d2d[:, j] for j in range(d2d.shape[1])]
        f3 = self.compile(ind[lo.idx_l3], self.psets.l3)
        pred = _vec(f3(*cols3), N)
        return pred


def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if not np.all(np.isfinite(pred)):
        return 1e6
    pred = np.clip(pred, -50.0, 50.0)
    err = pred - target
    val = float(np.sqrt(np.mean(err * err)))
    if not np.isfinite(val):
        return 1e6
    return val


# =============================================================================
# Init + seeding (prior ensemble-mean cho L2/L3)
# =============================================================================

def _mean_tree(pset: gp.PrimitiveSet) -> gp.PrimitiveTree:
    """Cây tính mean các argument của pset = mul(sum, 1/n)."""
    add = pset.mapping["add"]
    mul = pset.mapping["mul"]
    args = [pset.mapping[a] for a in pset.arguments]
    n = len(args)
    const = gp.Terminal(1.0 / n, False, float)
    nodes = [mul]
    for i in range(n - 1):
        nodes.append(add)
        nodes.append(args[i])
    nodes.append(args[n - 1])
    nodes.append(const)
    return gp.PrimitiveTree(nodes)


def _init_individual(cfg: GPConfig, layout: Layout, psets: PSets,
                     seeded: bool) -> list:
    trees = [None] * layout.total
    for i in range(cfg.K):
        for j in range(cfg.q):
            ps = psets.by_pos[layout.l1(i, j)]
            expr = gp.genHalfAndHalf(ps, min_=cfg.init_l1[0], max_=cfg.init_l1[1])
            trees[layout.l1(i, j)] = gp.PrimitiveTree(expr)
        # L2
        if seeded:
            trees[layout.l2(i)] = _mean_tree(psets.l2)
        else:
            expr = gp.genHalfAndHalf(psets.l2, min_=cfg.init_l23[0], max_=cfg.init_l23[1])
            trees[layout.l2(i)] = gp.PrimitiveTree(expr)
    # L3
    if seeded:
        trees[layout.idx_l3] = _mean_tree(psets.l3)  # mean(s_0..s_{K-1}, desc2d...)
    else:
        expr = gp.genHalfAndHalf(psets.l3, min_=cfg.init_l23[0], max_=cfg.init_l23[1])
        trees[layout.idx_l3] = gp.PrimitiveTree(expr)
    return trees


# =============================================================================
# Operators: crossover / mutation role-respecting + per-layer staticLimit
# =============================================================================

_LAYER_WEIGHTS = {L1: 0.5, L2: 0.3, L3: 0.2}


def _pick_positions(layout: Layout, allowed: List[int], n_min=1, n_max=3) -> List[int]:
    """Rút 1–3 vị trí: chọn lớp theo trọng số rồi chọn vị trí trong lớp."""
    layers = [l for l in (L1, L2, L3) if l in allowed]
    weights = [_LAYER_WEIGHTS[l] for l in layers]
    k = random.randint(n_min, min(n_max, layout.total))
    chosen = set()
    tries = 0
    while len(chosen) < k and tries < 50:
        tries += 1
        layer = random.choices(layers, weights=weights, k=1)[0]
        pos = random.choice(layout.positions(layer))
        chosen.add(pos)
    return list(chosen)


def _height_limit(cfg: GPConfig, layout: Layout, k: int) -> int:
    layer = layout.layer_of(k)
    return (cfg.height_l1 if layer == L1 else
            cfg.height_l2 if layer == L2 else cfg.height_l3)


class Operators:
    def __init__(self, cfg: GPConfig, layout: Layout, psets: PSets):
        self.cfg = cfg
        self.layout = layout
        self.psets = psets

    def mate(self, ind1, ind2, allowed: List[int]):
        for k in _pick_positions(self.layout, allowed):
            b1, b2 = gp.PrimitiveTree(ind1[k]), gp.PrimitiveTree(ind2[k])
            gp.cxOnePoint(ind1[k], ind2[k])
            lim = _height_limit(self.cfg, self.layout, k)
            if ind1[k].height > lim:
                ind1[k] = b1
            if ind2[k].height > lim:
                ind2[k] = b2
        del ind1.fitness.values
        del ind2.fitness.values
        return ind1, ind2

    def mutate(self, ind, allowed: List[int]):
        for k in _pick_positions(self.layout, allowed):
            ps = self.psets.by_pos[k]
            backup = gp.PrimitiveTree(ind[k])
            op = random.choice(("uniform", "node", "ephemeral", "shrink"))
            try:
                if op == "uniform":
                    def expr(pset, type_=None):
                        return gp.genFull(pset, min_=0, max_=2, type_=type_)
                    ind[k], = gp.mutUniform(ind[k], expr=expr, pset=ps)
                elif op == "node":
                    ind[k], = gp.mutNodeReplacement(ind[k], pset=ps)
                elif op == "ephemeral":
                    ind[k], = gp.mutEphemeral(ind[k], mode="one")
                else:
                    ind[k], = gp.mutShrink(ind[k])
            except (IndexError, ValueError):
                ind[k] = backup
            if ind[k].height > _height_limit(self.cfg, self.layout, k):
                ind[k] = backup
        del ind.fitness.values
        return ind,


# =============================================================================
# Driver
# =============================================================================

def _ensure_creator():
    if not hasattr(creator, "FitnessMinGP"):
        creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualGP"):
        creator.create("IndividualGP", list, fitness=creator.FitnessMinGP)


def _allowed_layers(warmup_active: bool) -> List[int]:
    return [L1] if warmup_active else [L1, L2, L3]


@dataclass
class GPResult:
    best_rmse_std_val: float
    test_rmse: float
    val_rmse: float
    train_rmse: float
    best_individual: list
    history: List[dict] = field(default_factory=list)


def run_gp(cache: Dict[str, object], cfg: GPConfig, verbose: bool = True) -> GPResult:
    """Chạy DEAP GP head trên 1 feature cache (đã standardize).

    cache: dict với key 'train'/'valid'/'test' (mỗi cái có conf_emb/desc3d/desc2d/
    target/target_raw) + 'stats' (để denormalize).
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _ensure_creator()

    tr, va, te = cache["train"], cache["valid"], cache["test"]
    stats = cache["stats"]
    n3d = tr["desc3d"].shape[2]
    cfg.num_2d = tr["desc2d"].shape[1]  # đồng bộ với số desc2d đã chọn lúc extract

    # Subspace cố định (seed được): mỗi slot sample d chiều từ 128 (cho phép chồng lấn).
    rng = np.random.default_rng(cfg.seed)
    emb_dim = tr["conf_emb"].shape[2]
    subspaces = [np.sort(rng.choice(emb_dim, size=cfg.d, replace=False))
                 for _ in range(cfg.num_emb)]

    layout = Layout(cfg)
    psets = PSets(cfg, layout, subspaces, n3d)
    compiler = Compiler()
    fwd = Forward(cfg, layout, psets, subspaces, compiler)
    ops = Operators(cfg, layout, psets)

    def make_arrays(split):
        return (split["conf_emb"].astype(np.float64),
                split["desc3d"].astype(np.float64),
                split["desc2d"].astype(np.float64),
                split["target"].astype(np.float64))

    tr_emb, tr_d3, tr_d2, tr_y = make_arrays(tr)
    va_emb, va_d3, va_d2, va_y = make_arrays(va)
    te_emb, te_d3, te_d2, te_y = make_arrays(te)

    def eval_ind(ind) -> Tuple[float]:
        pred = fwd.predict(ind, tr_emb, tr_d3, tr_d2)
        return (_rmse(pred, tr_y),)

    toolbox = base.Toolbox()
    toolbox.register("select", tools.selDoubleTournament,
                     fitness_size=cfg.tourn_fitness_size,
                     parsimony_size=cfg.tourn_parsimony, fitness_first=True)

    # --- Khởi tạo quần thể ---
    pop = []
    n_seed = int(round(cfg.seed_prob * cfg.pop_size))
    for idx in range(cfg.pop_size):
        trees = _init_individual(cfg, layout, psets, seeded=(idx < n_seed) or cfg.warmup > 0)
        ind = creator.IndividualGP(trees)
        pop.append(ind)
    # Khi warmup>0: TẤT CẢ cá thể khởi tạo L2/L3 = mean-combiner (đã set ở trên).

    for ind in pop:
        ind.fitness.values = eval_ind(ind)

    hof = tools.HallOfFame(1)
    hof.update(pop)

    def val_rmse_of(ind) -> float:
        return _rmse(fwd.predict(ind, va_emb, va_d3, va_d2), va_y)

    best_val = val_rmse_of(hof[0])
    best_ind = hof[0]
    history = []

    mu, lam = cfg.pop_size, cfg.pop_size
    for gen in range(1, cfg.generations + 1):
        warm = gen <= cfg.warmup
        allowed = _allowed_layers(warm)

        # varOr với gating warmup
        offspring = []
        while len(offspring) < lam:
            r = random.random()
            if r < cfg.cxpb:
                a, b = (toolbox.clone(x) for x in random.sample(pop, 2))
                a, _ = ops.mate(a, b, allowed)
                offspring.append(a)
            elif r < cfg.cxpb + cfg.mutpb:
                a = toolbox.clone(random.choice(pop))
                a, = ops.mutate(a, allowed)
                offspring.append(a)
            else:
                offspring.append(toolbox.clone(random.choice(pop)))

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = eval_ind(ind)

        pop[:] = toolbox.select(pop + offspring, mu)
        hof.update(pop)

        v = val_rmse_of(hof[0])
        if v < best_val:
            best_val = v
            best_ind = toolbox.clone(hof[0])

        if verbose and (gen % 5 == 0 or gen == 1):
            tr_best = hof[0].fitness.values[0]
            phase = "L1only" if warm else "joint"
            print(f"  gen {gen:3d} [{phase}] | train_rmse(std)={tr_best:.4f} | "
                  f"val_rmse(std)={v:.4f} | best_val={best_val:.4f}")
        history.append({"gen": gen, "train_std": hof[0].fitness.values[0],
                        "val_std": v})

    # Báo cáo: denormalize về đơn vị gốc (RMSE scale theo target_std).
    ts = stats.target_std
    train_std = _rmse(fwd.predict(best_ind, tr_emb, tr_d3, tr_d2), tr_y)
    val_std = _rmse(fwd.predict(best_ind, va_emb, va_d3, va_d2), va_y)
    test_std = _rmse(fwd.predict(best_ind, te_emb, te_d3, te_d2), te_y)

    return GPResult(
        best_rmse_std_val=best_val,
        test_rmse=test_std * ts,
        val_rmse=val_std * ts,
        train_rmse=train_std * ts,
        best_individual=best_ind,
        history=history,
    )
