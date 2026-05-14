"""Simulated Bifurcation (SB) シミュレータ — aSB / bSB / dSB の 3 バリアント。

参考:
- Goto, Tatsumura, Dixon, "Combinatorial optimization by simulating adiabatic
  bifurcations in nonlinear Hamiltonian systems", Science Advances 5,
  eaav2372 (2019).
- Goto et al., "High-performance combinatorial optimization based on classical
  mechanics", Science Advances 7, eabe7953 (2021).

========================================================================
【SB の物理的イメージ】
N 個のスピンを N 個の古典振動子 (x_i, y_i) で表し、Hamilton 力学に従って
時間発展させる。分岐パラメータ a(t) を 0 → a0 に徐々に増やすことで、
x = 0 の単一平衡点が分岐して x = ±x* の双安定状態に分かれる。
分岐の方向が結合 J_ij の影響を受けて Ising 基底状態に対応する配置に
収束する仕組み。
========================================================================

【方程式 (2021 paper Eq. (1)-(3))】
  共通の x の更新: x_i(t+dt) = x_i(t) + dt * a0 * y_i(t+dt)  (symplectic Euler)

  aSB (adiabatic SB):
    dy/dt = -[(a0 - a(t)) + K x_i^2] x_i + c0 Σ_j J_ij x_j
    Kerr 非線形項 K x^3 が振動を抑え、断熱的に基底状態へ
    壁拘束なし(滑らかな力学)

  bSB (ballistic SB):
    dy/dt = -(a0 - a(t)) x_i + c0 Σ_j J_ij x_j
    Kerr 項なし、代わりに |x|=1 で反射する壁拘束
    壁: |x_i| > 1 なら x_i ← sign(x_i), y_i ← 0

  dSB (discrete SB):
    dy/dt = -(a0 - a(t)) x_i + c0 Σ_j J_ij sign(x_j)
    bSB と同じだが、結合に sign(x_j) を使う(離散化)
    精度が向上

【パラメータ (推奨値)】
  a0       = 1.0
  dt       = 0.5
  N_step   = 1000 (ベンチマーク標準)
  c0       = 0.5 √((N-1) / Σ J_ij²)  (Goto 2021 推奨)
  K        = 1.0  (aSB のみ)
"""
import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix


# ============================================================
#  Numba JIT 版の本体 (3 バリアント共通の枠で variant フラグで分岐)
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_sb_batch(
    n: int,
    num_steps: int,
    num_trials: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_w: np.ndarray,    # 各辺の MAX-CUT 重み(unweighted の場合 +1)
    a0: float,
    c0: float,
    dt: float,
    K: float,
    variant: int,  # 0=aSB, 1=bSB, 2=dSB, 3=HbSB, 4=HdSB
    init_scale: float,
    gamma_heat: float,  # Heated SB の加熱率 (HbSB / HdSB のみ使用)
    seeds: np.ndarray,
):
    """trial 並列の SB 計算。variant で 5 つの式を切り替える。

    cut 値は edge_w を用いた重み付きで計算する。unweighted のときは edge_w=+1。
    """
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)
    num_edges = edge_a.shape[0]

    # ---- フラグ展開 ----
    has_kerr = (variant == 0)                       # aSB のみ Kerr 項
    has_wall = (variant >= 1)                       # aSB 以外は壁拘束
    use_sign = (variant == 2) or (variant == 4)     # dSB, HdSB は sign(x_j)
    has_heat = (variant >= 3)                       # HbSB, HdSB は加熱項

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        # 初期化: 小さな乱数
        x = np.empty(n, dtype=np.float64)
        y = np.empty(n, dtype=np.float64)
        for i in range(n):
            x[i] = (np.random.random() - 0.5) * 2.0 * init_scale
            y[i] = (np.random.random() - 0.5) * 2.0 * init_scale

        Jx = np.zeros(n, dtype=np.float64)
        y_old = np.zeros(n, dtype=np.float64)       # 加熱項用に y(t_k) を保存
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18  # 重み付き cut なので負の値もあり得る → 強い負で初期化

        for k in range(num_steps):
            # 分岐パラメータの線形ランプ: a(t) ∈ [0, a0]
            a_t = (k + 1) / num_steps * a0
            a_diff = a0 - a_t  # (a0 - a(t))

            # ---- 加熱用に y(t_k) を保存 ----
            if has_heat:
                for i in range(n):
                    y_old[i] = y[i]

            # ---- 結合項 J @ x (CSR 手書きループ) ----
            if use_sign:
                # dSB, HdSB: sign(x_j) を使う
                for i in range(n):
                    acc = 0.0
                    start = J_indptr[i]
                    end = J_indptr[i + 1]
                    for jj in range(start, end):
                        j = J_indices[jj]
                        sj = 1.0 if x[j] > 0.0 else -1.0
                        acc += J_data[jj] * sj
                    Jx[i] = acc
            else:
                # aSB, bSB, HbSB: x_j そのまま
                for i in range(n):
                    acc = 0.0
                    start = J_indptr[i]
                    end = J_indptr[i + 1]
                    for jj in range(start, end):
                        acc += J_data[jj] * x[J_indices[jj]]
                    Jx[i] = acc

            # ---- y の更新 ----
            if has_kerr:
                # aSB: Kerr 項 K x^3 を含む
                for i in range(n):
                    force = -(a_diff + K * x[i] * x[i]) * x[i] + c0 * Jx[i]
                    y[i] = y[i] + dt * force
            else:
                # bSB, dSB, HbSB, HdSB: Kerr 項なし
                for i in range(n):
                    force = -a_diff * x[i] + c0 * Jx[i]
                    y[i] = y[i] + dt * force

            # ---- x の更新 (symplectic Euler) ----
            for i in range(n):
                x[i] = x[i] + dt * a0 * y[i]

            # ---- 壁拘束 (bSB, dSB, HbSB, HdSB) ----
            if has_wall:
                for i in range(n):
                    if x[i] > 1.0:
                        x[i] = 1.0
                        y[i] = 0.0
                    elif x[i] < -1.0:
                        x[i] = -1.0
                        y[i] = 0.0

            # ---- 加熱項 (HbSB, HdSB のみ): y(t_{k+1}) = ỹ̃ + γ y(t_k) Δt ----
            # 論文 Eq.(22): 壁拘束の「後」に γ y(t_k) Δt を加える。
            # こうすることで、壁衝突で y=0 になっても再び動き出せる。
            if has_heat:
                for i in range(n):
                    y[i] = y[i] + gamma_heat * y_old[i] * dt

            # ---- cut 評価 (重み付き) ----
            cut = 0.0
            for e in range(num_edges):
                if (x[edge_a[e]] > 0.0) != (x[edge_b[e]] > 0.0):
                    cut += edge_w[e]

            if cut > best_cut:
                best_cut = cut
                for i in range(n):
                    best_signs[i] = x[i] > 0.0

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out


# ============================================================
#  Python 側公開ラッパー
# ============================================================
VARIANT_ID = {"aSB": 0, "bSB": 1, "dSB": 2, "HbSB": 3, "HdSB": 4}


def auto_c0(J: csr_matrix, n: int) -> float:
    """Goto 2021 推奨の c0: 0.5 √((N-1) / Σ J_ij²)。

    J が対称 (上下三角に同じ値を持つ) ことを前提に、全要素の二乗和で計算。
    """
    sum_J2 = float((J.data ** 2).sum())
    if sum_J2 <= 0.0:
        return 0.0
    return 0.5 * np.sqrt((n - 1) / sum_J2)


def simulate_sb_batch(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_steps: int,
    num_trials: int,
    *,
    variant: str = "bSB",
    a0: float = 1.0,
    c0: float | None = None,
    dt: float = 0.5,
    K: float = 1.0,
    init_scale: float = 0.1,
    gamma_heat: float = 0.0,
    weights: list[float] | None = None,
    seeds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SB を num_trials 並列実行する。

    Args:
        variant:    "aSB", "bSB", "dSB", "HbSB", "HdSB" のいずれか
        c0:         None なら auto_c0(J, n) を使用
        gamma_heat: HbSB / HdSB の加熱率(論文推奨: HbSB γ=0.5, HdSB γ=0.06)
        weights:    各辺の MAX-CUT 重み(K2000 のような ±1 重み付きグラフ用)。
                    None なら +1 を仮定し従来の unweighted カット数を返す。
        seeds:      各 trial の seed 配列。None なら np.arange(num_trials)

    Returns:
        best_cuts:  shape (num_trials,) の float (重み付き cut 値の最大)
        best_signs: shape (num_trials, n) の bool
    """
    if variant not in VARIANT_ID:
        raise ValueError(f"variant must be one of {list(VARIANT_ID)}, got {variant!r}")
    if c0 is None:
        c0 = auto_c0(J, n)
    if seeds is None:
        seeds = np.arange(num_trials, dtype=np.int64)

    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    best_cuts, best_signs = _simulate_sb_batch(
        n, num_steps, num_trials,
        J.data, J.indices, J.indptr,
        edge_a, edge_b, edge_w,
        float(a0), float(c0), float(dt), float(K),
        VARIANT_ID[variant],
        float(init_scale),
        float(gamma_heat),
        seeds_arr,
    )
    return best_cuts, best_signs
