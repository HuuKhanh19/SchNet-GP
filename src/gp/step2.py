"""STEP 2 — đo sức mạnh của descriptor 2D (KHÔNG dùng embedding).

Học thẳng desc2d (toàn bộ ~217 RDKit 2D, không chọn) -> target, 2 mode:
  - "ridge": RidgeCV (sklearn) — co hệ số (L2) tự giảm trọng số descriptor vô dụng.
  - "tree" : MỘT cây GP symbolic (DEAP) — biến không được tham chiếu = bị loại.

Cả hai chia sẻ cùng tiền xử lý:
  inf -> NaN -> điền median TRAIN -> z-score theo TRAIN -> bỏ cột hằng (std~0 trên
  train) -> clip z vào [-clip, clip] (ghìm outlier descriptor như Ipc).

RMSE báo cáo ở ĐƠN VỊ GỐC của target (so trực tiếp baseline SchNet 1-conf
0.8994 ± 0.0946). Ridge fit trên y gốc; cây GP học y standardized rồi denormalize.
"""

from __future__ import annotations

import operator
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, gp, tools

from .gp_head import _OPS, FUNCSET_L1, _vec

EPS = 1e-8


# =============================================================================
# Tiền xử lý: clean + standardize (fit TRAIN, chống leak)
# =============================================================================

@dataclass
class Standardizer:
    fill: np.ndarray      # median TRAIN mỗi cột GỐC (điền NaN/inf)
    mean: np.ndarray      # mean TRAIN sau khi điền (cột giữ lại)
    std: np.ndarray       # std  TRAIN (cột giữ lại)
    keep: np.ndarray      # bool (D,) cột không hằng trên train
    names: List[str]      # tên cột GIỮ LẠI (sau khi bỏ cột hằng)
    clip: float
    target_mean: float
    target_std: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64).copy()
        X[~np.isfinite(X)] = np.nan
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(self.fill, nan_idx[1])
        X = X[:, self.keep]
        Z = (X - self.mean) / self.std
        return np.clip(Z, -self.clip, self.clip)


def fit_standardizer(Xtr: np.ndarray, ytr: np.ndarray, names: List[str],
                     clip: float = 10.0) -> Standardizer:
    X = Xtr.astype(np.float64).copy()
    X[~np.isfinite(X)] = np.nan
    fill = np.nanmedian(X, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(fill, nan_idx[1])

    mean_all = X.mean(0)
    std_all = X.std(0)
    keep = std_all > EPS                       # bỏ cột hằng trên train (vô dụng)
    kept_names = [n for n, k in zip(names, keep) if k]
    return Standardizer(
        fill=fill, mean=mean_all[keep], std=std_all[keep] + EPS,
        keep=keep, names=kept_names, clip=clip,
        target_mean=float(ytr.mean()), target_std=float(ytr.std()) + EPS,
    )


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


# =============================================================================
# MODE 1 — RidgeCV
# =============================================================================

@dataclass
class RidgeResult:
    train_rmse: float
    val_rmse: float
    test_rmse: float
    alpha: float
    top_coef: List[Tuple[str, float]] = field(default_factory=list)


def run_ridge(Xtr, ytr, Xva, yva, Xte, yte, names: List[str],
              alphas: Optional[np.ndarray] = None, top_k: int = 15) -> RidgeResult:
    """RidgeCV trên desc2d standardized (y giữ đơn vị gốc)."""
    from sklearn.linear_model import RidgeCV

    if alphas is None:
        alphas = np.logspace(-3, 4, 22)
    model = RidgeCV(alphas=alphas)
    model.fit(Xtr, ytr)
    order = np.argsort(-np.abs(model.coef_))[:top_k]
    return RidgeResult(
        train_rmse=_rmse(model.predict(Xtr), ytr),
        val_rmse=_rmse(model.predict(Xva), yva),
        test_rmse=_rmse(model.predict(Xte), yte),
        alpha=float(model.alpha_),
        top_coef=[(names[i], float(model.coef_[i])) for i in order],
    )


# =============================================================================
# MODE 2 — một cây GP symbolic (DEAP)
# =============================================================================

@dataclass
class GPTreeConfig:
    pop_size: int = 500
    generations: int = 120
    mu: int = 500
    lam: int = 500
    cxpb: float = 0.7
    mutpb: float = 0.25
    height: int = 17            # full-coverage = cây TO -> cần cao hơn
    max_len: int = 6000         # trần số node (vượt thì revert) tránh phình vô hạn
    init_min: int = 2
    init_max: int = 5
    tourn_fitness_size: int = 5
    tourn_parsimony: float = 1.4
    full_coverage: bool = True  # True: MỘT cây to dùng MỌI biến desc2d (repair coverage),
                                # không để cây tự chọn lọc biến. False: cây tự do (cũ).
    seed: int = 0


@dataclass
class GPTreeResult:
    train_rmse: float
    val_rmse: float
    test_rmse: float
    best_val_std: float
    formula: str
    size: int
    used_descriptors: List[str] = field(default_factory=list)
    history: List[dict] = field(default_factory=list)


_ERC_N = [0]


def _erc():
    return random.gauss(0.0, 1.0) if random.random() < 0.5 else random.uniform(-2.0, 2.0)


def _build_pset(n_in: int) -> gp.PrimitiveSet:
    ps = gp.PrimitiveSet("D2D", n_in)
    ps.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_in)})
    for nm in FUNCSET_L1:
        fn, ar = _OPS[nm]
        ps.addPrimitive(fn, ar, name=nm)
    _ERC_N[0] += 1
    ps.addEphemeralConstant(f"erc_d2d_{_ERC_N[0]}", _erc)
    return ps


# --- Full-coverage: một cây TO dùng MỌI biến desc2d --------------------------

def _combine_add(subtrees: List[list], add_prim) -> list:
    """Gộp list subtree (mỗi cái = list node prefix) thành 1 cây add CÂN BẰNG
    (height ~log, không bị sâu tuyến tính khi gom nhiều biến)."""
    nodes = [list(s) for s in subtrees if s]
    if not nodes:
        return []
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                nxt.append([add_prim] + nodes[i] + nodes[i + 1])
            else:
                nxt.append(nodes[i])
        nodes = nxt
    return nodes[0]


def _full_tree_nodes(pset: gp.PrimitiveSet, n_in: int, add_prim) -> list:
    """Cây khởi tạo phủ TẤT CẢ biến x_0..x_{n_in-1}: mỗi biến bọc ngẫu nhiên 1 unary
    và/hoặc nhân hằng (trọng số) rồi cộng cân bằng -> một cây to, MỌI biến có mặt."""
    unary = [pset.mapping[n] for n in ("square", "psqrt", "abs", "neg", "tanh", "plog")]
    mul = pset.mapping["mul"]
    subs = []
    for i in range(n_in):
        leaf = [pset.mapping[f"x{i}"]]
        if random.random() < 0.5:
            leaf = [random.choice(unary)] + leaf
        if random.random() < 0.5:
            leaf = [mul, gp.Terminal(_erc(), False, float)] + leaf
        subs.append(leaf)
    # Scale về trung bình (mul 1/n) -> output ~O(1) thay vì ~O(sqrt(n)), dễ tối ưu hơn.
    return [mul, gp.Terminal(1.0 / max(n_in, 1), False, float)] + _combine_add(subs, add_prim)


def _ensure_creator():
    if not hasattr(creator, "FitnessMinD2D"):
        creator.create("FitnessMinD2D", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualD2D"):
        creator.create("IndividualD2D", gp.PrimitiveTree, fitness=creator.FitnessMinD2D)


def run_gp_tree(Xtr, ytr_z, Xva, yva_raw, Xte, yte_raw,
                target_mean: float, target_std: float, names: List[str],
                cfg: GPTreeConfig, verbose: bool = True) -> GPTreeResult:
    """MỘT cây GP học desc2d (standardized) -> target (standardized), chọn best theo val.

    full_coverage=True (mặc định): cây TO, đầu vào là MỌI biến desc2d. Khởi tạo phủ hết
    biến; sau mỗi lai/đột biến REPAIR -> biến nào bị rớt được ghép lại bằng add cân bằng,
    nên cây KHÔNG tự chọn lọc biến. Chọn lọc bằng tournament fitness (không ép parsimony
    co cây); trần height/len để khỏi phình vô hạn.
    full_coverage=False: cây tự do, tự chọn biến (chống bloat bằng double tournament).
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    _ensure_creator()

    n_in = Xtr.shape[1]
    pset = _build_pset(n_in)
    add_prim = pset.mapping["add"]
    trcols = [Xtr[:, i] for i in range(n_in)]
    vacols = [Xva[:, i] for i in range(n_in)]
    tecols = [Xte[:, i] for i in range(n_in)]

    _cache: Dict[str, object] = {}

    def compile_(ind):
        key = str(ind)
        f = _cache.get(key)
        if f is None:
            f = gp.compile(ind, pset)
            _cache[key] = f
        return f

    def evaluate(ind):
        pred = _vec(compile_(ind)(*trcols), len(ytr_z))
        # fitness trên thang standardized; cắt outlier để RMSE không nổ vô hạn.
        pred = np.clip(np.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6), -50, 50)
        return (_rmse(pred, ytr_z),)

    def rmse_raw(ind, cols, y_raw):
        pred = _vec(compile_(ind)(*cols), len(y_raw)) * target_std + target_mean
        return _rmse(pred, y_raw)

    def expr_mut(pset, type_=None):              # DEAP gọi expr(pset=..., type_=...)
        return gp.genFull(pset, min_=0, max_=2, type_=type_)

    def mutate(ind, full):
        saved = creator.IndividualD2D(ind)        # bản sao đúng kiểu (fallback khi op lỗi)
        r = random.random()
        try:
            if full:                              # full-coverage: chỉ thay subtree/đổi node
                if r < 0.5:                       # (repair phía sau lo phủ lại biến)
                    return gp.mutUniform(ind, expr=expr_mut, pset=pset)
                return gp.mutNodeReplacement(ind, pset=pset)
            if r < 0.55:
                return gp.mutUniform(ind, expr=expr_mut, pset=pset)
            if r < 0.78:
                return gp.mutNodeReplacement(ind, pset=pset)
            if r < 0.9:
                return gp.mutEphemeral(ind, mode="one")
            return gp.mutShrink(ind)
        except (IndexError, ValueError):
            return (saved,)

    def repair(ind):
        """Ghép lại biến bị thiếu -> MỌI biến desc2d luôn có mặt trong cây."""
        present = {int(m) for m in re.findall(r"\bx(\d+)\b", str(ind))}
        missing = [i for i in range(n_in) if i not in present]
        if not missing:
            return ind
        miss = _combine_add([[pset.mapping[f"x{i}"]] for i in missing], add_prim)
        # Ghép biến thiếu như số hạng nhỏ (mul 1/n) để không làm nổ biên độ cây hiện có.
        scaled = [pset.mapping["mul"], gp.Terminal(1.0 / n_in, False, float)] + miss
        return creator.IndividualD2D([add_prim] + list(ind) + scaled)

    def within(ind):
        return ind.height <= cfg.height and len(ind) <= cfg.max_len

    toolbox = base.Toolbox()
    clone = toolbox.clone

    if cfg.full_coverage:
        pop = [creator.IndividualD2D(_full_tree_nodes(pset, n_in, add_prim))
               for _ in range(cfg.pop_size)]

        def select(items, k):
            return tools.selTournament(items, k, tournsize=cfg.tourn_fitness_size)

        def make_offspring():
            off = []
            while len(off) < cfg.lam:
                r = random.random()
                if r < cfg.cxpb and len(pop) >= 2:
                    p1, p2 = random.sample(pop, 2)
                    a, b = clone(p1), clone(p2)
                    gp.cxOnePoint(a, b)
                    cand = repair(a)
                elif r < cfg.cxpb + cfg.mutpb:
                    p1 = random.choice(pop)
                    a, = mutate(clone(p1), full=True)
                    cand = repair(a)
                else:
                    off.append(clone(random.choice(pop)))   # sinh sản: giữ nguyên fitness
                    continue
                if cand.fitness.valid:                       # cây đã đổi -> đánh giá lại
                    del cand.fitness.values
                if not within(cand):                         # vượt trần -> revert về cha
                    cand = clone(p1)
                off.append(cand)
            return off
    else:
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                         min_=cfg.init_min, max_=cfg.init_max)
        toolbox.register("individual", tools.initIterate, creator.IndividualD2D, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", lambda ind: mutate(ind, full=False))
        hlim = gp.staticLimit(key=operator.attrgetter("height"), max_value=cfg.height)
        toolbox.decorate("mate", hlim)
        toolbox.decorate("mutate", hlim)
        pop = toolbox.population(n=cfg.pop_size)

        def select(items, k):
            return tools.selDoubleTournament(
                items, k, fitness_size=cfg.tourn_fitness_size,
                parsimony_size=cfg.tourn_parsimony, fitness_first=True)

        def make_offspring():
            return algorithms.varOr(pop, toolbox, cfg.lam, cfg.cxpb, cfg.mutpb)

    for ind in pop:
        ind.fitness.values = evaluate(ind)
    hof = tools.HallOfFame(1)
    hof.update(pop)

    best_val = rmse_raw(hof[0], vacols, yva_raw)
    best_ind = clone(hof[0])
    best_val_std = hof[0].fitness.values[0]
    history = []

    for gen in range(1, cfg.generations + 1):
        offspring = make_offspring()
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)
        pop[:] = select(pop + offspring, cfg.mu)
        hof.update(pop)

        v = rmse_raw(hof[0], vacols, yva_raw)
        if v < best_val:
            best_val = v
            best_ind = clone(hof[0])
            best_val_std = hof[0].fitness.values[0]
        history.append({"gen": gen, "train_std": hof[0].fitness.values[0], "val_raw": v})
        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"    gen {gen:3d} | train_rmse(std)={hof[0].fitness.values[0]:.4f} | "
                  f"val_rmse(raw)={v:.4f} | best_val={best_val:.4f} | size={len(hof[0])}")

    formula = str(best_ind)
    used_idx = sorted({int(m) for m in re.findall(r"\bx(\d+)\b", formula)})
    return GPTreeResult(
        train_rmse=rmse_raw(best_ind, trcols, ytr_z * target_std + target_mean),
        val_rmse=best_val,
        test_rmse=rmse_raw(best_ind, tecols, yte_raw),
        best_val_std=best_val_std,
        formula=formula,
        size=len(best_ind),
        used_descriptors=[names[i] for i in used_idx],
        history=history,
    )
