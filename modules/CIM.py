"""
Coherent Ising Machine (CIM) シミュレータ - 進行波モデル

論文: Inoue & Yoshida,
  "Traveling-wave model of coherent Ising machine based on fiber loop with
   pulse-pumped phase-sensitive amplifier",
  Optics Communications 522 (2022) 128642.

========================================================================
【CIM の物理的イメージ】
ファイバーのループ(周回路)の中を、m個の光パルスが順番に周回している。
ループの途中には PSA(位相感応増幅器) があり、パルスが通過するたびに
「in-phase 成分(c_i)」は増幅、「quadrature 成分(s_i)」は減衰する。
さらに MFB(測定フィードバック) によって、パルス同士は結合行列 J_ij を通じて
互いに影響し合う。
最終的に各パルスの in-phase 振幅は正 or 負 の 2値に収束し、
その符号が Ising スピン(= 0/1 割り当て)を表す。
MAX-CUT 問題は J_ij を辺として与え、反強磁性結合(J < 0)にすることで、
「異なるクラスタに属するペアの数の最大化」に対応する。
========================================================================

【主要パラメータ (論文 Section 3)】
  κ = 130 W^(-1/2) m^(-1)  非線形定数(PSAの利得係数を決める)
  L = 5 cm                  PSA媒質長
  γ = 42.09 W^(-1)          飽和係数(信号が強いほど利得が飽和)
  loop loss = 11 dB         ループ全体の損失 → η = 10^(-1.1) ≈ 0.0794
  BW = 1 GHz                システム帯域(ノイズ分散 Eq.6 に乗算)
  dP/round = 0.05 mW        毎ラウンドでポンプパワーを増やす量
  J_ij ∈ {0, -0.03}         G22 の結合係数(辺なら -0.03、なければ 0)
"""

import numpy as np
import wandb
from numba import njit, prange
from scipy.sparse import csr_matrix
from scipy.sparse._sparsetools import csr_matvec

from .verify import compute_cut_from_edges, run_all_checks


# ============================================================
#  Numba JIT 版のコアループ (wandb_log=False 時に使用)
# ============================================================
# 内ループ全体を ahead-of-time コンパイルしてネイティブコード化する。
# - numpy の ufunc ディスパッチオーバーヘッドを全部スキップ
# - scipy.sparse のラッパー呼び出しもバイパス
# - 1 パルスあたりの演算を融合ループ化して中間配列のアロケーションも削減
# 初回は JIT コンパイルに数秒かかるが、2回目以降は cache (.numba_cache) から復元される。
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_batch(
    n: int,
    num_rounds: int,
    num_trials: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_w: np.ndarray,    # MAX-CUT 重み(未指定なら +1 で埋める)
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seeds: np.ndarray,
):
    """CIM シミュレーションを num_trials 分まとめて並列実行する(Numba JIT)。

    prange により trial 単位で CPU コアに分散。
    各 trial は独立なので競合なく並列化できる。
    num_trials=1 で呼べば単発実行にもなる。
    cut は edge_w 重み付き(unweighted の場合 edge_w=+1)。
    """
    # 出力バッファ
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)

    # 事前計算可能な定数(全 trial 共通)
    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]

    # ---- trial ごとに並列実行 ----
    for trial_idx in prange(num_trials):
        # 各 thread で独立な乱数状態(numba は thread-local RNG)
        np.random.seed(seeds[trial_idx])

        # trial ごとに独立な状態ベクトル
        c = np.zeros(n, dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18

        for k in range(num_rounds):
            # Step 1: ポンプパワー → 非飽和利得係数 g_0
            P_p = (k + 1) * dP_per_round
            g0 = 2.0 * kappa * np.sqrt(P_p) * L
            half_g0 = 0.5 * g0
            neg_half_g0_gamma = -0.5 * g0 * gamma

            # Step 2: Sparse matvec Jc = J @ c (CSR 手書きループ)
            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * c[J_indices[jj]]
                Jc[i] = acc

            # Step 3-5: coupled_in → I_in → sqrt(G_I) → noise → c を 1ループで融合
            for i in range(n):
                coupled_in_i = sqrt_eta * c[i] + Jc[i]
                I_in_i = coupled_in_i * coupled_in_i
                half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
                sqrt_G_I_i = np.exp(half_g_i)
                noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                c[i] = sqrt_G_I_i * coupled_in_i + noise_i

            # Step 6: 重み付き cut 計算
            cut = 0.0
            for e in range(num_edges):
                if (c[edge_a[e]] > 0.0) != (c[edge_b[e]] > 0.0):
                    cut += edge_w[e]

            # ベスト更新
            if cut > best_cut:
                best_cut = cut
                for i in range(n):
                    best_signs[i] = c[i] > 0.0

        # 結果を出力バッファに格納
        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out


def load_graph(filepath: str, return_weights: bool = False):
    """Gset 形式のグラフを読み込み、隣接リストと辺リストを返す。

    入力ファイルは 1-indexed だが、Python では配列を 0-indexed で扱いたいので
    ここで -1 して変換している。

    辺の 3 番目の列があれば重み w_ij として解釈する。無ければ +1 とみなす。
    K2000 のような ±1 重み付きインスタンスでは return_weights=True で重みも取得。
    """
    with open(filepath, "r") as f:
        n, k = map(int, f.readline().split())
        adj = [[] for _ in range(n)]
        edges = []
        weights: list[float] = []
        for _ in range(k):
            parts = f.readline().split()
            a, b = int(parts[0]) - 1, int(parts[1]) - 1
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            adj[a].append(b)
            adj[b].append(a)
            edges.append((a, b))
            weights.append(w)
    if return_weights:
        return n, k, adj, edges, weights
    return n, k, adj, edges


def build_coupling_matrix(
    n: int,
    edges: list[tuple[int, int]],
    coupling: float,
    weights: list[float] | None = None,
) -> csr_matrix:
    """結合行列 J を CSR 形式のスパース行列として構築する。

    通常モード (weights=None): すべての辺で J_ij = coupling(従来挙動)。
      G22 (w=+1) なら coupling=-0.03 で J_ij = -0.03 (反強磁性、カット促進)。

    重み付きモード (weights 指定): J_ij = coupling * w_ij。
      K2000 (w ∈ ±1) なら coupling=-1 で J_ij ∈ ±1 になり、論文の SB 設定と一致。
      w=+1 の辺 → J=-|coupling| (cut 促進)、w=-1 の辺 → J=+|coupling| (cut 抑制)。
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    if weights is None:
        for a, b in edges:
            rows.append(a); cols.append(b); data.append(coupling)
            rows.append(b); cols.append(a); data.append(coupling)
    else:
        for (a, b), w in zip(edges, weights):
            val = coupling * w
            rows.append(a); cols.append(b); data.append(val)
            rows.append(b); cols.append(a); data.append(val)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def amplitudes_to_solution(c: np.ndarray) -> list[int]:
    """in-phase 振幅 c_i の符号から 0/1 の割り当てを決定する。

    c_i > 0  → 集合 B (= 1)
    c_i ≤ 0  → 集合 A (= 0)

    注: 全体の符号を反転しても MAX-CUT のカット数は同じなので、
    どちらを 0/1 に割り当てるかは任意。
    """
    return [1 if ci > 0 else 0 for ci in c]


def simulate_cim(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_rounds: int,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    rng: np.random.Generator,
    log_interval: int = 10,
    wandb_log: bool = True,
) -> tuple[np.ndarray, int, list[int]]:
    """進行波モデルによる CIM シミュレーションのメインループ。

    ==== 使用する式 (論文より) ====
    Eq.(3):  c_i(k+1) = √G_I,i · (√η · c_i(k) + Σ_j J_ij c_j(k)) + N_I,i
             s_i(k+1) = √(η·G_Q,i) · s_i(k) + N_Q,i
             ↑ 1ラウンド分の発展方程式。
                √η は ループ損失(信号が弱くなる)
                √G_I は PSA 増幅(in-phase を増幅)
                √G_Q は PSA 減衰(quadrature を減衰)
                Σ_j J_ij c_j は MFB による他パルスからの結合入力
                N_I, N_Q は増幅器からの ASE + 真空ノイズ

    Eq.(14): g   = g_0·(1 - γ·I_in)      ← 飽和込みの利得係数
             G_I = exp(g)                  ← in-phase 利得
             G_Q = exp(-g)                 ← quadrature 利得(減衰)
             g_0(k) = 2·κ·√(P_p(k))·L     ← 非飽和時の利得係数

    Eq.(15): I_in ≈ (√η·c_i + Σ_j J_ij·c_j)^2
             ↑ PSA への入力信号強度(飽和項 γ·I_in に使う)
                vacuum ノイズと s_i は強度が小さいので無視している

    Eq.(6):  σ²_I = (2-η)·G_I/4 · BW      ← in-phase ノイズの分散
             σ²_Q = (2-η)·G_Q/4 · BW      ← quadrature ノイズの分散
    """

    # ---- 辺配列の事前計算(cut 評価の高速化) ----
    # 辺リストを numpy 配列化。
    edges_np = np.asarray(edges, dtype=np.int64)  # shape: (K, 2)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    edge_w = np.ones(edges_np.shape[0], dtype=np.float64)  # unweighted

    # ====== Fast path: wandb 出力が不要なら Numba JIT 版を直接呼ぶ ======
    # 内部では _simulate_cim_batch を num_trials=1 で呼び、単発実行にも対応。
    if not wandb_log:
        seed = int(rng.integers(0, 2**63 - 1))
        seeds = np.array([seed], dtype=np.int64)
        best_cuts_out, best_signs_out = _simulate_cim_batch(
            n,
            num_rounds,
            1,
            J.data,
            J.indices,
            J.indptr,
            edge_a,
            edge_b,
            edge_w,
            float(kappa),
            float(L),
            float(gamma),
            float(eta),
            float(bandwidth),
            float(photon_energy),
            float(dP_per_round),
            seeds,
        )
        best_cut = int(best_cuts_out[0])
        best_x: list[int] = best_signs_out[0].astype(np.int64).tolist()
        # c_final は返さない(JIT 内で持っていないため、ゼロベクトルでダミー)
        return np.zeros(n, dtype=np.float64), best_cut, best_x

    # ====== Slow path: wandb 出力あり (単発実行・デバッグ用) ======
    # ---- 初期条件 ----
    # 全パルスは vacuum 状態から始まる → c(0) = 0
    # 最初のノイズ N_I によって自発的に立ち上がっていく。
    # ※ quadrature 成分 s_i は coupled_in や cut 計算に一切関与しないため、
    #   計算を省略(論文 Eq.3b は形式上存在するが、simulation の結果に影響しない)。
    c = np.zeros(n, dtype=np.float64)

    # ---- 事前計算可能な定数 ----
    # ノイズ σ²_I = (2-η)·G_I/4·BW·ℏω → σ_I = noise_const * √G_I と分解できる。
    # 定数部分を1回だけ計算しておき、毎ラウンド sqrt を呼ばない。
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    sqrt_eta = np.sqrt(eta)
    half_dP = 0.5  # sqrt(G_I) = exp(g/2) の係数

    # ---- scipy sparse のラッパーオーバーヘッドを回避 ----
    # 毎ラウンド J @ c を呼ぶと 1500×(250μs wrapper + 170μs 計算) = 630ms かかる。
    # 内部の低レベル csr_matvec を直接呼ぶことでラッパー部分をスキップする。
    # 注意: csr_matvec は「加算」なので、呼ぶ前に出力バッファを必ず 0 にする必要がある。
    J_data = J.data
    J_indices = J.indices
    J_indptr = J.indptr
    Jc = np.zeros(n, dtype=np.float64)

    # ベストカット数と、その時の符号配列を追跡
    best_cut = 0
    best_signs = np.zeros(n, dtype=bool)

    # ---- メインループ ----
    for k in range(num_rounds):
        # Step 1: ポンプパワー → 非飽和利得係数 g_0 (毎ラウンド更新)
        P_p = (k + 1) * dP_per_round
        g0 = 2.0 * kappa * np.sqrt(P_p) * L

        # Step 2: J @ c をバッファに直接書き込む (wrapper オーバーヘッド回避)
        # csr_matvec は accumulator なので事前に 0 クリアが必要
        Jc.fill(0.0)
        csr_matvec(n, n, J_indptr, J_indices, J_data, c, Jc)
        # coupled_in = √η·c + J·c  (Eq.15)
        coupled_in = sqrt_eta * c + Jc
        I_in = coupled_in * coupled_in

        # Step 3: 利得を計算 (Eq.14)
        # g = g_0·(1 - γ·I_in), sqrt(G_I) = exp(g/2) として直接計算。
        # G_I 自体は不要(σ_I も sqrt(G_I) から出せる)。
        half_g = 0.5 * g0 * (1.0 - gamma * I_in)
        sqrt_G_I = np.exp(half_g)

        # Step 4: ノイズ生成 σ_I = noise_const * sqrt(G_I)
        # rng.standard_normal + 乗算 は rng.normal(0, σ) と同速だがシンプル
        N_I = rng.standard_normal(n) * (noise_const * sqrt_G_I)

        # Step 5: 差分方程式 Eq.3a で振幅更新
        # c(k+1) = sqrt(G_I)·coupled_in + N_I
        c = sqrt_G_I * coupled_in + N_I

        # Step 6: cut 評価 (ベクトル化 + .sum() メソッドで wrapper 回避)
        signs = c > 0
        cut = int((signs[edge_a] != signs[edge_b]).sum())
        if cut > best_cut:
            best_cut = cut
            best_signs = signs.copy()

        # Step 7: wandb ログ (wandb_log=False 時は完全スキップ)
        if wandb_log and ((k + 1) % log_interval == 0 or k == 0):
            # ログ用にだけ G_I と sigma_I を復元
            G_I = sqrt_G_I * sqrt_G_I
            sigma_I = noise_const * sqrt_G_I
            mean_abs_c = float(np.mean(np.abs(c)))
            mean_sigma = float(sigma_I.mean())
            wandb.log({
                "round": k + 1,
                "pump_power_mW": P_p * 1e3,
                "g0": g0,
                "eta_G_I_unsat": eta * float(np.exp(2.0 * half_dP * g0)),
                "mean_abs_c": mean_abs_c,
                "std_c": float(c.std()),
                "mean_I_in": float(I_in.mean()),
                "mean_G_I": float(G_I.mean()),
                "mean_sigma_I": mean_sigma,
                "snr": mean_abs_c / (mean_sigma + 1e-30),
                "current_cut": cut,
                "best_cut": best_cut,
            })

    # 最後に一度だけ bool → int list に変換して返す
    best_x: list[int] = best_signs.astype(np.int64).tolist()
    return c, best_cut, best_x


def simulate_cim_batch(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_rounds: int,
    num_trials: int,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seeds: np.ndarray,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """100 trial を並列実行するための公開ラッパー。

    weights が指定された場合、cut は重み付きで計算される。

    Returns:
        best_cuts: shape (num_trials,)      各 trial の最良カット値
        best_signs: shape (num_trials, n)   各 trial の最良解(bool)
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    best_cuts, best_signs = _simulate_cim_batch(
        n,
        num_rounds,
        num_trials,
        J.data,
        J.indices,
        J.indptr,
        edge_a,
        edge_b,
        edge_w,
        float(kappa),
        float(L),
        float(gamma),
        float(eta),
        float(bandwidth),
        float(photon_energy),
        float(dP_per_round),
        seeds_arr,
    )
    return best_cuts, best_signs


def main():
    # ==== ハイパーパラメータ設定 ====
    # 論文 Section 3 の値をそのまま使用
    config = {
        # 物理パラメータ
        "kappa": 130.0,             # W^(-1/2) m^(-1)  非線形定数
        "L": 0.05,                  # m                PSA長 (5 cm)
        "gamma": 42.09,             # W^(-1)           飽和係数
        "loss_dB": 11.0,            # dB               ループ損失
        "bandwidth": 1.0e9,         # Hz               システム帯域 (1 GHz)
        "photon_energy_J": 1.28e-19, # J               一光子エネルギー ℏω (1550nm)
                                     #                  ノイズ式の単位変換因子
        "dP_per_round": 0.05e-3,    # W/round          ポンプ増加量 (0.05 mW)
        "coupling": -0.03,          #                  G22 辺の結合係数

        # シミュレーション設定
        "num_rounds": 1500,         # 総ラウンド数
        "seed": 42,                 # 乱数シード
        "log_interval": 10,         # wandb ログ間隔
    }

    # ==== wandb 初期化 ====
    wandb.init(project="cim-max-cut", config=config)
    cfg = wandb.config

    # 損失(dB) → 透過率 η に変換
    #   loss_dB = 11 → η = 10^(-11/10) ≈ 0.0794
    #   ループを1周するたびに信号パワーは η 倍になる
    eta = 10.0 ** (-cfg.loss_dB / 10.0)
    wandb.config.update({"eta": eta}, allow_val_change=True)

    # 乱数生成器(seed 固定で再現性を確保)
    rng = np.random.default_rng(cfg.seed)

    # ==== グラフ読み込み ====
    filepath = "input/G22.txt"
    n, k_edges, adj, edges = load_graph(filepath)
    print(f"N={n}, K={k_edges}, eta={eta:.4f}")

    # ==== 結合行列の構築 ====
    # G22 の辺に対して J_ij = -0.03 を設定
    J = build_coupling_matrix(n, edges, cfg.coupling)

    # ==== CIM シミュレーション実行 ====
    print("Running CIM simulation...")
    c_final, best_cut, best_x = simulate_cim(
        n=n,
        J=J,
        edges=edges,
        num_rounds=cfg.num_rounds,
        kappa=cfg.kappa,
        L=cfg.L,
        gamma=cfg.gamma,
        eta=eta,
        bandwidth=cfg.bandwidth,
        photon_energy=cfg.photon_energy_J,
        dP_per_round=cfg.dP_per_round,
        rng=rng,
        log_interval=cfg.log_interval,
    )

    # ==== 結果表示 ====
    print(f"Best cut: {best_cut}")
    print(f"Paper (Fig.8 G22): mean=13275, best=13321")
    print(f"Known best: 13359")

    # ==== 検算 (scripts/verify.py の独立実装と突き合わせ) ====
    run_all_checks(best_x, n, k_edges, adj, edges, best_cut)

    # ==== wandb summary に結果を記録 ====
    wandb.summary["best_cut"] = best_cut
    wandb.summary["ratio_to_known_best"] = best_cut / 13359
    wandb.summary["paper_mean"] = 13275
    wandb.summary["paper_best"] = 13321

    wandb.finish()


if __name__ == "__main__":
    main()
