"""共通通信ライブラリ (comm.py)

第1〜4週のPythonプログラミング演習で繰り返し使う「光通信の基本部品」をまとめた
モジュール。各問題の solution.py は、このファイルを import して使う。

含まれる機能:
  - PRBS (擬似ランダムビット列) の生成               generate_prbs
  - グレイ符号化した方形QAMのマッピング/デマッピング   bits_to_symbols / symbols_to_bits
  - 硬判定 (最近傍シンボル)                          decide
  - SNR指定でのAWGN (白色ガウス雑音) 付加             add_awgn
  - 方形QAMのBER解析解 (近似式)                       ber_theory_qam
  - 規格化定数・コンステレーション点の取得             qam_norm / qam_constellation
  - レイズドコサインフィルタ                          rrc_filter / rc_filter

設計方針:
  - すべて「平均シンボル電力 = 1」に規格化した複素ベースバンド表現で統一する。
  - グレイ符号は I軸 (実部) と Q軸 (虚部) を独立な √M 値PAMとして扱い、
    各軸でグレイ符号化する (方形QAMの標準的な構成)。
  - ビット配置: 1シンボル log2(M) ビットのうち、前半 log2(√M) ビットをI軸、
    後半 log2(√M) ビットをQ軸に割り当てる (どちらも MSB first のグレイ符号)。
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# PRBS (Pseudo Random Bit Sequence)
# =============================================================================
def generate_prbs(n_bits: int = 15, initial_state=None) -> np.ndarray:
    """最大長 LFSR (M系列) で PRBS を 1 周期分生成する。

    生成多項式 x^15 + x^14 + 1 (タップ b13, b14) の Fibonacci 型 LFSR。
    周期は 2**n_bits - 1。第1週の問1で詳しく扱う。

    Args:
        n_bits: シフトレジスタのビット数 (既定 15)。
        initial_state: b0..b_{n-1} の初期値。省略時は全ビット 1。

    Returns:
        長さ 2**n_bits - 1 の 0/1 ビット列 (np.int8)。
    """
    period = 2 ** n_bits - 1
    if initial_state is None:
        reg = [1] * n_bits
    else:
        reg = list(initial_state)
        if len(reg) != n_bits:
            raise ValueError("初期状態のビット数が n_bits と一致しません")
        if all(b == 0 for b in reg):
            raise ValueError("初期状態を全 0 にすると系列が生成されません")

    out = np.empty(period, dtype=np.int8)
    for i in range(period):
        out[i] = reg[-1]                     # 最高次ビットを出力
        feedback = reg[-2] ^ reg[-1]         # b13 XOR b14
        reg = [feedback] + reg[:-1]          # 右シフトし b0 に帰還
    return out


# =============================================================================
# グレイ符号ユーティリティ (整数 <-> グレイ符号)
# =============================================================================
def _gray_encode(n: np.ndarray) -> np.ndarray:
    """通常の2進整数 -> グレイ符号 (n XOR (n>>1))。"""
    return n ^ (n >> 1)


def _gray_decode(g: np.ndarray, nbits: int) -> np.ndarray:
    """グレイ符号 -> 通常の2進整数 (b = g ^ (g>>1) ^ (g>>2) ^ ...)。"""
    b = g.copy()
    for s in range(1, nbits):
        b ^= (g >> s)
    return b


def _int_to_bits(vals: np.ndarray, nbits: int) -> np.ndarray:
    """整数配列 -> (len, nbits) のビット行列 (MSB first)。"""
    shifts = np.arange(nbits - 1, -1, -1)
    return ((vals[:, None] >> shifts) & 1).astype(np.int8)


def _bits_to_int(bits: np.ndarray) -> np.ndarray:
    """(len, nbits) のビット行列 (MSB first) -> 整数配列。"""
    nbits = bits.shape[1]
    weights = (1 << np.arange(nbits - 1, -1, -1))
    return bits.astype(np.int64) @ weights


# =============================================================================
# 方形QAM
# =============================================================================
def qam_params(M: int):
    """(k=1シンボルのビット数, L=√M=軸あたりレベル数, kk=軸あたりビット数) を返す。"""
    k = int(round(np.log2(M)))
    L = int(round(np.sqrt(M)))
    if L * L != M or (k % 2) != 0:
        raise ValueError(f"M={M} は方形QAM (4,16,64,256,...) ではありません")
    return k, L, k // 2


def qam_norm(M: int) -> float:
    """平均シンボル電力を 1 に規格化するための割り算係数 sqrt(2(M-1)/3)。

    レベル {±1, ±3, ..., ±(L-1)} の方形QAMの平均電力は 2(M-1)/3。
    """
    return np.sqrt(2.0 * (M - 1) / 3.0)


def qam_constellation(M: int) -> np.ndarray:
    """規格化済みの全 M 個のコンステレーション点を返す (シンボル番号順ではない)。"""
    _, L, _ = qam_params(M)
    levels = 2 * np.arange(L) - (L - 1)           # {-(L-1),...,(L-1)}
    I, Q = np.meshgrid(levels, levels)
    pts = (I.ravel() + 1j * Q.ravel()) / qam_norm(M)
    return pts


def bits_to_symbols(bits: np.ndarray, M: int) -> np.ndarray:
    """ビット列 -> 規格化複素QAMシンボル列 (QAMマッピング)。

    bits の長さは log2(M) の倍数であること。前半 kk ビット=I軸グレイ符号、
    後半 kk ビット=Q軸グレイ符号として、各軸を √M 値PAMにマップする。
    """
    k, L, kk = qam_params(M)
    bits = np.asarray(bits, dtype=np.int8)
    if bits.size % k != 0:
        raise ValueError(f"ビット数 {bits.size} が log2(M)={k} の倍数ではありません")
    g = bits.reshape(-1, k)

    i_gray = _bits_to_int(g[:, :kk])               # I軸グレイ符号 (整数)
    q_gray = _bits_to_int(g[:, kk:])               # Q軸グレイ符号 (整数)
    i_idx = _gray_decode(i_gray, kk)               # 0..L-1 (振幅順)
    q_idx = _gray_decode(q_gray, kk)

    i_amp = 2 * i_idx - (L - 1)
    q_amp = 2 * q_idx - (L - 1)
    return (i_amp + 1j * q_amp) / qam_norm(M)


def _slice_indices(sym: np.ndarray, M: int):
    """規格化シンボル -> 各軸の最近傍レベル番号 (0..L-1) を返す (硬判定の中核)。"""
    _, L, _ = qam_params(M)
    x = sym * qam_norm(M)                          # 整数振幅スケールに戻す
    i_idx = np.clip(np.rint((x.real + (L - 1)) / 2.0), 0, L - 1).astype(np.int64)
    q_idx = np.clip(np.rint((x.imag + (L - 1)) / 2.0), 0, L - 1).astype(np.int64)
    return i_idx, q_idx


def decide(sym: np.ndarray, M: int) -> np.ndarray:
    """硬判定: 各サンプルを最も近いコンステレーション点へ丸める。"""
    _, L, _ = qam_params(M)
    i_idx, q_idx = _slice_indices(sym, M)
    i_amp = 2 * i_idx - (L - 1)
    q_amp = 2 * q_idx - (L - 1)
    return (i_amp + 1j * q_amp) / qam_norm(M)


def symbols_to_bits(sym: np.ndarray, M: int) -> np.ndarray:
    """シンボル列 -> ビット列 (硬判定 + QAMデマッピング)。

    入力が雑音を含む受信シンボルでも、内部で最近傍判定してからビットに戻す。
    """
    _, L, kk = qam_params(M)
    i_idx, q_idx = _slice_indices(sym, M)
    i_gray = _gray_encode(i_idx)
    q_gray = _gray_encode(q_idx)
    i_bits = _int_to_bits(i_gray, kk)
    q_bits = _int_to_bits(q_gray, kk)
    return np.concatenate([i_bits, q_bits], axis=1).ravel()


# =============================================================================
# 雑音とBER
# =============================================================================
def add_awgn(sym: np.ndarray, snr_db: float, signal_power: float | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """複素AWGN (白色ガウス雑音) を SNR [dB] 指定で付加する。

    SNR = (信号電力) / (雑音電力)。複素雑音なので実部・虚部に半分ずつ分配する。

    Args:
        sym: 複素ベースバンド信号。
        snr_db: 信号電力対雑音電力比 [dB]。
        signal_power: 信号電力。省略時は sym から実測 (平均 |sym|^2)。
        rng: 乱数生成器 (再現性のため指定可)。

    Returns:
        雑音付加後の複素信号。
    """
    if rng is None:
        rng = np.random.default_rng()
    if signal_power is None:
        signal_power = np.mean(np.abs(sym) ** 2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_lin           # 複素雑音の総電力 N0
    sigma = np.sqrt(noise_power / 2.0)             # 1軸あたりの標準偏差
    noise = sigma * (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape))
    return sym + noise


def count_bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> int:
    """送信/受信ビット列の不一致数。"""
    n = min(len(tx_bits), len(rx_bits))
    return int(np.count_nonzero(tx_bits[:n] != rx_bits[:n]))


def ber_theory_qam(M: int, snr_db, snr_per: str = "symbol") -> np.ndarray:
    """グレイ符号化 方形M-QAM の BER 解析解 (近似式) を返す。

        BER ≈ (4/k)(1 - 1/√M) Q( sqrt( 3/(M-1) · γ_s ) )

    γ_s は1シンボルあたりのSNR (Es/N0)。QPSKでは厳密解 Q(√γ_s) に一致する。

    Args:
        M: QAM多値数。
        snr_db: SNR [dB] (配列可)。
        snr_per: "symbol" なら γ_s=10^(snr/10)、"bit" なら γ_b として Es/N0=k·γ_b。
    """
    from math import erfc  # スカラ用。配列は scipy 非依存で自前 Q を使う。

    k, L, _ = qam_params(M)
    snr_db = np.asarray(snr_db, dtype=float)
    snr_lin = 10 ** (snr_db / 10.0)
    gamma_s = snr_lin if snr_per == "symbol" else snr_lin * k

    # Q(x) = 0.5 * erfc(x/sqrt(2)) を numpy で
    from numpy import vectorize
    qfunc = vectorize(lambda x: 0.5 * erfc(x / np.sqrt(2.0)))
    arg = np.sqrt(3.0 / (M - 1) * gamma_s)
    return (4.0 / k) * (1.0 - 1.0 / L) * qfunc(arg)


# =============================================================================
# 2x2 MIMO LMS 適応等化 (バタフライ構成)
# =============================================================================
def mimo_lms(rx: np.ndarray, ref: np.ndarray, ntaps: int, mu: float):
    """偏波多重信号の 2x2 MIMO LMS 等化 (データ援用)。

    バタフライ FIR:
        v_x[n] = Σ_j W_xx[j] u_x[n-..] + W_xy[j] u_y[n-..]
        v_y[n] = Σ_j W_yx[j] u_x[n-..] + W_yy[j] u_y[n-..]
    各出力の誤差 e = d - v で全タップを LMS 更新する。

    Args:
        rx:  受信両偏波信号 (2, N)。
        ref: 参照(送信)両偏波シンボル (2, N)。
        ntaps: タップ数 (1 なら単一タップ)。
        mu: ステップサイズ。

    Returns:
        (v_out, valid): v_out は等化出力 (2, N)、valid は有効サンプルの
        インデックス範囲 (slice)。中心タップ基準で参照を合わせている。
    """
    L = ntaps
    two, N = rx.shape
    c = L // 2                                 # 中心タップ位置
    W = np.zeros((2, 2, L), dtype=complex)
    W[0, 0, c] = 1.0                           # 中心タップを単位行列で初期化
    W[1, 1, c] = 1.0
    v_out = np.zeros((2, N), dtype=complex)

    for n in range(L - 1, N):
        ux = rx[0, n - L + 1:n + 1]            # 長さ L (古い→新しい)
        uy = rx[1, n - L + 1:n + 1]
        vx = np.dot(W[0, 0], ux) + np.dot(W[0, 1], uy)
        vy = np.dot(W[1, 0], ux) + np.dot(W[1, 1], uy)
        t = n - L + 1 + c                       # 中心タップが対応する時刻
        dx = ref[0, t]
        dy = ref[1, t]
        ex = dx - vx
        ey = dy - vy
        W[0, 0] += mu * ex * np.conj(ux)
        W[0, 1] += mu * ex * np.conj(uy)
        W[1, 0] += mu * ey * np.conj(ux)
        W[1, 1] += mu * ey * np.conj(uy)
        v_out[0, t] = vx
        v_out[1, t] = vy

    valid = slice(c, N - (L - 1 - c))
    return v_out, valid


def mimo_lms_fse(rx: np.ndarray, ref: np.ndarray, ntaps: int, mu: float,
                 sps: int, base: int):
    """分数間隔 (fractionally-spaced) 2x2 MIMO LMS 等化。

    入力をオーバーサンプリング (sps sample/symbol) のまま受け、シンボルレートで
    1 出力を出す (sps サンプルごとにスライド)。多タップにすると、整合フィルタ・
    タイミング補償・偏波分離をまとめて学習でき、サンプリング位相のずれに強くなる。

    出力シンボル k は入力窓 rx[:, k*sps+base : k*sps+base+ntaps] を使う。

    Args:
        rx:  受信両偏波信号 (2, Nsamp)。
        ref: 参照(送信)両偏波シンボル (2, Nsym)。
        ntaps: タップ数。
        mu: ステップサイズ。
        sps: 1シンボルあたりサンプル数。
        base: k=0 のときの入力開始インデックス (群遅延・タイミング位相を含む)。

    Returns:
        (v_out, kmax): v_out は等化出力 (2, Nsym)、kmax は有効な最大シンボル番号。
    """
    L = ntaps
    Nsym = ref.shape[1]
    Nsamp = rx.shape[1]
    W = np.zeros((2, 2, L), dtype=complex)
    W[0, 0, L // 2] = 1.0
    W[1, 1, L // 2] = 1.0
    v_out = np.zeros((2, Nsym), dtype=complex)
    kmax = min((Nsamp - base - L) // sps, Nsym - 1)
    k0 = max(0, (-base + sps - 1) // sps)
    for k in range(k0, kmax + 1):
        i0 = k * sps + base
        ux = rx[0, i0:i0 + L]
        uy = rx[1, i0:i0 + L]
        vx = np.dot(W[0, 0], ux) + np.dot(W[0, 1], uy)
        vy = np.dot(W[1, 0], ux) + np.dot(W[1, 1], uy)
        ex = ref[0, k] - vx
        ey = ref[1, k] - vy
        W[0, 0] += mu * ex * np.conj(ux)
        W[0, 1] += mu * ex * np.conj(uy)
        W[1, 0] += mu * ey * np.conj(ux)
        W[1, 1] += mu * ey * np.conj(uy)
        v_out[0, k] = vx
        v_out[1, k] = vy
    return v_out, kmax


# =============================================================================
# レーザ位相雑音 (ウィーナー過程)
# =============================================================================
def laser_phase_noise(n: int, df: float, fs: float,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """ローレンツ型線幅 df [Hz] のレーザ位相雑音 θ[n] を生成する (ランダムウォーク)。

    位相増分は分散 σ_PN^2 = 2π·df/fs の AWGN。最初のサンプルは -π〜π の一様乱数。

        θ[0] = uniform(-π, π),   θ[n] = θ[n-1] + w[n],  w[n] ~ N(0, 2π·df/fs)

    Args:
        n:  サンプル数。
        df: スペクトル線幅 (FWHM) [Hz]。
        fs: サンプリングレート [Hz]。
        rng: 乱数生成器。

    Returns:
        長さ n の位相系列 [rad]。
    """
    if rng is None:
        rng = np.random.default_rng()
    var = 2.0 * np.pi * df / fs                     # 位相増分の分散
    incr = rng.standard_normal(n) * np.sqrt(var)
    incr[0] = rng.uniform(-np.pi, np.pi)            # 初期位相は一様分布
    return np.cumsum(incr)


# =============================================================================
# レイズドコサイン (パルス整形) フィルタ
# =============================================================================
def rc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """レイズドコサイン (RC) フィルタのインパルス応答。

    Args:
        beta: ロールオフ係数 α (0〜1)。
        sps:  1シンボルあたりサンプル数 (oversampling factor)。
        span: フィルタ長 (シンボル数)。全長 = span*sps+1。

    Returns:
        ピークが 1 に規格化された対称FIR係数。
    """
    n = np.arange(-span * sps / 2, span * sps / 2 + 1)
    t = n / sps                                    # シンボル時間単位
    sinc = np.sinc(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
    # 特異点 (1-(2βt)^2 = 0) を極限値で埋める
    if beta > 0:
        sing = np.isclose(np.abs(2 * beta * t), 1.0)
        cos[sing] = np.pi / 4
    h = sinc * cos
    return h


def pulse_shape(sym: np.ndarray, sps: int, beta: float, span: int):
    """シンボル列を raised cosine で整形する (0挿入アップサンプル + 畳み込み)。

    Returns:
        (shaped, delay): shaped は整形サンプル列、delay は群遅延 [サンプル]。
        shaped[delay + n*sps] が n 番目のシンボル中心 (ISIなしのタイミング)。
    """
    h = rc_filter(beta, sps, span)
    up = np.zeros(len(sym) * sps, dtype=complex)
    up[::sps] = sym
    shaped = np.convolve(up, h)
    delay = (len(h) - 1) // 2
    return shaped, delay


def downsample_isi_free(shaped: np.ndarray, delay: int, sps: int, n_sym: int) -> np.ndarray:
    """整形サンプル列から、ISIが生じないシンボル中心タイミングを抽出する。"""
    idx = delay + np.arange(n_sym) * sps
    idx = idx[idx < len(shaped)]
    return shaped[idx]


def rrc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """ルートレイズドコサイン (RRC) フィルタのインパルス応答 (エネルギー規格化)。"""
    N = span * sps
    n = np.arange(-N / 2, N / 2 + 1)
    t = n / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


# =============================================================================
# 自己テスト
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=== comm.py 自己テスト ===")

    # 1) PRBS の平衡性
    for nb in (7, 15):
        seq = generate_prbs(nb)
        ones = int(seq.sum())
        print(f"PRBS{nb}: 周期={len(seq)}, 1の数={ones} (理論 {2**(nb-1)})")
        assert len(seq) == 2 ** nb - 1 and ones == 2 ** (nb - 1)

    # 2) QAM マッピングの往復 (無雑音なら完全復元)
    for M in (4, 16, 64, 256):
        k, L, kk = qam_params(M)
        nsym = 5000
        bits = rng.integers(0, 2, size=nsym * k).astype(np.int8)
        sym = bits_to_symbols(bits, M)
        # コンステレーション全体の平均電力は厳密に 1、ランダム標本は近似的に 1
        p_exact = np.mean(np.abs(qam_constellation(M)) ** 2)
        p = np.mean(np.abs(sym) ** 2)
        rx_bits = symbols_to_bits(sym, M)
        ok = np.array_equal(bits, rx_bits)
        print(f"M={M:3d}: 規格化電力(厳密)={p_exact:.6f}, 標本電力={p:.4f}, 無雑音往復一致={ok}")
        assert ok and abs(p_exact - 1.0) < 1e-12 and abs(p - 1.0) < 0.05

    # 3) 雑音時のBERが解析解とおおむね一致するか (16QAM, SNR=15dB)
    M = 16
    k = qam_params(M)[0]
    nsym = 200000
    bits = rng.integers(0, 2, size=nsym * k).astype(np.int8)
    sym = bits_to_symbols(bits, M)
    snr = 15.0
    rx = add_awgn(sym, snr, rng=rng)
    rx_bits = symbols_to_bits(rx, M)
    ber = count_bit_errors(bits, rx_bits) / len(bits)
    ber_th = float(ber_theory_qam(M, snr))
    print(f"16QAM SNR=15dB: 実測BER={ber:.3e}, 解析解={ber_th:.3e}")
    assert 0.3 < ber / ber_th < 3.0

    print("すべてのテストに合格しました。")
