"""GP head đa-cây (DEAP) — Phase 3.

Cá thể = list cây (PrimitiveTree). Forward phân tầng, vectorize trên cả tập phân tử:

    bin i (conformer hạng energy i):
        f_{i,j} = feature_tree_j( terminal của slot j tại bin i )      # q cây
        s_i     = combiner( f_{i,0..q-1} )
    pred       = global_tree( s_0..s_{K-1}, desc2d_0..desc2d_{m-1} )

Cờ tree_sharing điều khiển cây nào share giữa các bin (=> kích thước genotype):
    none     : mỗi bin bộ (q+1) cây riêng        -> K*(q+1) + 1
    features : q cây feature share, combiner/bin  -> q + K + 1
    all      : (q+1) cây share, chỉ global riêng  -> (q+1) + 1

Subspace (chiều embedding / 3D-descriptor mỗi slot) CỐ ĐỊNH (xem subspace.py);
cờ chỉ đổi việc share biểu thức, không đổi gán slot -> clean ablation.
"""

import functools
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from deap import base, creator, gp, tools

from .fitness import OPERATORS, fitness_value, _finite
from .subspace import Subspace

# Bộ đếm toàn cục để đặt tên ephemeral/ pset duy nhất (DEAP lưu vào globals của gp,
# tái đăng ký trùng tên sẽ lỗi khi chạy lưới nhiều cell trong cùng tiến trình).
_UID = [0]


def _next_uid() -> int:
    _UID[0] += 1
    return _UID[0]


def _ephemeral_uniform(lo: float, hi: float) -> float:
    """Ephemeral constant dùng global random (đã seed trong evolve) -> tái lập.

    Hàm module-level + functools.partial -> picklable, tránh cảnh báo lambda của DEAP.
    """
    return random.uniform(lo, hi)


def _ensure_creator():
    """Tạo FitnessMin + Individual một lần (idempotent)."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        # Cá thể = list cây; fitness gắn riêng.
        creator.create("Individual", list, fitness=creator.FitnessMin)


# =============================================================================
# Primitive sets
# =============================================================================

def _build_pset(arity: int, operators: List[str], ephemeral: Tuple[float, float],
                rng: random.Random) -> gp.PrimitiveSet:
    """Một PrimitiveSet float với `arity` input + operators + ephemeral constant."""
    tag = _next_uid()
    pset = gp.PrimitiveSet(f"PS{tag}", arity)
    for op_name in operators:
        if op_name not in OPERATORS:
            raise ValueError(f"Operator không hỗ trợ: {op_name}")
        fn, ar = OPERATORS[op_name]
        pset.addPrimitive(fn, ar, name=f"{op_name}_{tag}")
    lo, hi = ephemeral
    # Ephemeral name phải duy nhất toàn cục; partial(module-fn) -> picklable, không
    # bị DEAP cảnh báo lambda. rng giữ trong chữ ký cho tương thích (không dùng ở đây).
    pset.addEphemeralConstant(f"eph{tag}", functools.partial(_ephemeral_uniform, lo, hi))
    return pset


@dataclass
class GPLayout:
    """Bố cục genotype + psets theo cờ tree_sharing."""
    mode: str
    K: int
    q: int
    num_2d: int
    psets: List[gp.PrimitiveSet]      # 1 pset / vị trí cây trong genotype
    n_trees: int

    # offset truy cập theo mode (điền lúc build)
    _feat_index: Callable[[int, int], int] = field(default=None, repr=False)
    _comb_index: Callable[[int], int] = field(default=None, repr=False)
    _global_index: int = field(default=-1, repr=False)

    def feat_tree(self, funcs, i, j):
        return funcs[self._feat_index(i, j)]

    def comb_tree(self, funcs, i):
        return funcs[self._comb_index(i)]

    def global_tree(self, funcs):
        return funcs[self._global_index]


def build_layout(mode: str, K: int, subspace: Subspace, num_2d: int,
                 operators: List[str], ephemeral: Tuple[float, float],
                 rng: random.Random) -> GPLayout:
    """Dựng psets + index map cho 1 trong 3 cờ tree_sharing."""
    q = subspace.q

    # psets feature theo slot (arity = số chiều slot). emb slot arity = emb_dim;
    # desc3d slot arity = số 3D descriptor của slot. Tạo 1 pset / slot.
    feat_psets = [
        _build_pset(subspace.slot_dim(j), operators, ephemeral, rng)
        for j in range(q)
    ]
    comb_pset_factory = lambda: _build_pset(q, operators, ephemeral, rng)
    global_pset = _build_pset(K + num_2d, operators, ephemeral, rng)

    psets: List[gp.PrimitiveSet] = []

    if mode == "none":
        # [bin0: feat0..feat_{q-1}, comb0][bin1: ...] ... [global]
        for _ in range(K):
            psets.extend(feat_psets)        # NB: cùng pset object cho mọi bin của slot j
            psets.append(comb_pset_factory())
        psets.append(global_pset)

        def feat_index(i, j):
            return i * (q + 1) + j

        def comb_index(i):
            return i * (q + 1) + q

        global_index = K * (q + 1)

    elif mode == "features":
        # [feat0..feat_{q-1}] [comb0..comb_{K-1}] [global]
        psets.extend(feat_psets)
        comb_psets = [comb_pset_factory() for _ in range(K)]
        psets.extend(comb_psets)
        psets.append(global_pset)

        def feat_index(i, j):
            return j                        # share mọi bin

        def comb_index(i):
            return q + i

        global_index = q + K

    elif mode == "all":
        # [feat0..feat_{q-1}, comb] [global]
        psets.extend(feat_psets)
        psets.append(comb_pset_factory())
        psets.append(global_pset)

        def feat_index(i, j):
            return j

        def comb_index(i):
            return q                        # combiner share mọi bin

        global_index = q + 1

    else:
        raise ValueError(f"tree_sharing không hợp lệ: {mode}")

    layout = GPLayout(
        mode=mode, K=K, q=q, num_2d=num_2d, psets=psets, n_trees=len(psets),
    )
    layout._feat_index = feat_index
    layout._comb_index = comb_index
    layout._global_index = global_index
    return layout


# =============================================================================
# Problem (dữ liệu 1 split, đã standardize theo train)
# =============================================================================

@dataclass
class GPProblem:
    emb: np.ndarray          # (N, K, hidden) đã standardize theo train
    desc3d: np.ndarray       # (N, K, n3d)   đã standardize theo train
    desc2d: np.ndarray       # (N, num_2d)   đã standardize theo train
    target_std: np.ndarray   # (N,) target standardized
    subspace: Subspace

    @property
    def N(self) -> int:
        return self.emb.shape[0]

    def slot_args(self, i: int, j: int) -> List[np.ndarray]:
        """Terminal (list vector (N,)) của cây feature slot j tại bin i."""
        ss = self.subspace
        if j < ss.num_emb_trees:
            dims = ss.emb_slots[j]
            return [self.emb[:, i, d] for d in dims]
        d3 = ss.desc3d_slots[j - ss.num_emb_trees]
        return [self.desc3d[:, i, idx] for idx in d3]

    def desc2d_cols(self) -> List[np.ndarray]:
        return [self.desc2d[:, t] for t in range(self.desc2d.shape[1])]


def _as_vec(val, n: int) -> np.ndarray:
    """Ép kết quả eval cây về vector (n,) hữu hạn (hằng số -> broadcast)."""
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    elif arr.shape[0] != n:
        arr = np.broadcast_to(arr, (n,)).copy()
    return _finite(arr)


def predict(individual, problem: GPProblem, layout: GPLayout) -> np.ndarray:
    """Forward phân tầng -> pred (N,) ở không gian target STANDARDIZED."""
    funcs = [gp.compile(tree, ps) for tree, ps in zip(individual, layout.psets)]
    n, K, q = problem.N, layout.K, layout.q

    s_list = []
    for i in range(K):
        feats = []
        for j in range(q):
            f = layout.feat_tree(funcs, i, j)
            feats.append(_as_vec(f(*problem.slot_args(i, j)), n))
        comb = layout.comb_tree(funcs, i)
        s_list.append(_as_vec(comb(*feats), n))

    g = layout.global_tree(funcs)
    pred = g(*(s_list + problem.desc2d_cols()))
    return _as_vec(pred, n)


def genotype_nodes(individual) -> int:
    return int(sum(len(tree) for tree in individual))


# =============================================================================
# Toolbox + vòng tiến hoá
# =============================================================================

def _make_toolbox(layout: GPLayout, gp_cfg: dict) -> base.Toolbox:
    _ensure_creator()
    tb = base.Toolbox()
    init_min = int(gp_cfg.get("init_min_depth", 1))
    init_max = int(gp_cfg.get("init_max_depth", 3))

    # generator biểu thức theo từng pset (dùng cho init + mutation)
    def make_tree(pset):
        return gp.PrimitiveTree(gp.genHalfAndHalf(pset, init_min, init_max))

    def init_individual():
        return creator.Individual([make_tree(ps) for ps in layout.psets])

    tb.register("individual", init_individual)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("select", tools.selTournament,
                tournsize=int(gp_cfg.get("tournament_size", 5)))
    tb._make_tree = make_tree   # dùng trong mutation
    return tb


def _crossover(c1, c2, max_height: int):
    """cxOnePoint từng vị trí (cùng pset). Revert vị trí vượt max_height."""
    for k in range(len(c1)):
        a, b = c1[k], c2[k]
        na, nb = gp.cxOnePoint(gp.PrimitiveTree(a), gp.PrimitiveTree(b))
        if na.height <= max_height:
            c1[k] = na
        if nb.height <= max_height:
            c2[k] = nb


def _mutate(ind, layout: GPLayout, make_tree, rng: random.Random, max_height: int):
    """mutUniform tại MỘT vị trí ngẫu nhiên (subtree thay bằng cây mới từ pset đó)."""
    k = rng.randrange(len(ind))
    pset = layout.psets[k]

    # DEAP mutUniform gọi expr(pset=pset, type_=type_) bằng keyword -> tên tham số
    # PHẢI là `pset` (và nhận type_). genHalfAndHalf trả list node thay subtree.
    def expr(pset, type_=None):
        return gp.genHalfAndHalf(pset, 0, 2)

    mutated, = gp.mutUniform(gp.PrimitiveTree(ind[k]), expr=expr, pset=pset)
    if mutated.height <= max_height:
        ind[k] = mutated


def evolve(problem: GPProblem, layout: GPLayout, gp_cfg: dict, seed: int
           ) -> Tuple[object, float, dict]:
    """Chạy GP. Trả (best_individual, best_train_fitness, stats_dict)."""
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    tb = _make_toolbox(layout, gp_cfg)
    pop_size = int(gp_cfg.get("population", 500))
    n_gen = int(gp_cfg.get("generations", 100))
    cxpb = float(gp_cfg.get("cx_prob", 0.7))
    mutpb = float(gp_cfg.get("mut_prob", 0.2))
    max_height = int(gp_cfg.get("max_height", 8))
    parsimony = float(gp_cfg.get("parsimony_coef", 1e-3))

    def eval_ind(ind):
        pred = predict(ind, problem, layout)
        return (fitness_value(pred, problem.target_std,
                              genotype_nodes(ind), parsimony),)

    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = eval_ind(ind)

    hof = tools.HallOfFame(int(gp_cfg.get("hall_of_fame", 1)))
    hof.update(pop)

    for _ in range(n_gen):
        offspring = [tb.clone(ind) for ind in tb.select(pop, len(pop))]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if rng.random() < cxpb:
                _crossover(c1, c2, max_height)
                del c1.fitness.values
                del c2.fitness.values

        for mut in offspring:
            if rng.random() < mutpb:
                _mutate(mut, layout, tb._make_tree, rng, max_height)
                if mut.fitness.valid:
                    del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = eval_ind(ind)

        # elitism: giữ lại hof
        offspring[0] = tb.clone(hof[0])
        pop = offspring
        hof.update(pop)

    best = hof[0]
    stats = {
        "genotype_tree_count": layout.n_trees,
        "avg_tree_size": float(np.mean([len(t) for t in best])),
        "best_train_fitness": float(best.fitness.values[0]),
    }
    return best, float(best.fitness.values[0]), stats
