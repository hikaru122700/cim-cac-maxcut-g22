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

    PRBS (Pseudo Random Bit Sequence) は「決まった手順で作るが統計的にはランダムに
    見える」ビット列で、通信路の評価 (BER 測定など) の試験信号として使う。
    実装には LFSR (線形帰還シフトレジスタ) を用いる。レジスタの一部ビットを
    XOR して先頭に帰還しながらシフトすることで、長い周期の系列を生成できる。

    生成多項式 x^15 + x^14 + 1 (タップ b13, b14) の Fibonacci 型 LFSR。
    この多項式は「原始多項式」なので系列は最大長 (M系列) になり、周期は
    2**n_bits - 1 (= 全 0 を除く全状態を一巡)。M系列は 1 周期内で 1 の数が
    0 の数より 1 だけ多い「平衡性」を持つ。第1週の問1で詳しく扱う。

    Args:
        n_bits: シフトレジスタのビット数 (既定 15)。
        initial_state: b0..b_{n-1} の初期値 (長さ n_bits の 0/1 列)。
            省略時は全ビット 1。全 0 は禁止 (系列が動かないため)。

    Returns:
        長さ period = 2**n_bits - 1 の 0/1 ビット列 (shape=(period,), dtype=np.int8)。
    """
    period = 2 ** n_bits - 1                  # M系列の周期 (全 0 状態を除く)
    if initial_state is None:
        reg = [1] * n_bits                    # 既定の初期状態は全ビット 1
    else:
        reg = list(initial_state)
        if len(reg) != n_bits:
            raise ValueError("初期状態のビット数が n_bits と一致しません")
        if all(b == 0 for b in reg):
            # 全 0 状態は帰還しても 0 のまま固定され、系列が生成されない
            raise ValueError("初期状態を全 0 にすると系列が生成されません")

    out = np.empty(period, dtype=np.int8)
    for i in range(period):
        out[i] = reg[-1]                     # 最高次ビット (b14) を出力ビットとする
        feedback = reg[-2] ^ reg[-1]         # 帰還ビット = b13 XOR b14 (タップの XOR)
        reg = [feedback] + reg[:-1]          # 全体を右へ 1 シフトし、帰還ビットを b0 へ
    return out


# =============================================================================
# グレイ符号ユーティリティ (整数 <-> グレイ符号)
# =============================================================================
def _gray_encode(n: np.ndarray) -> np.ndarray:
    """通常の2進整数 -> グレイ符号。

    グレイ符号は「隣り合う数どうしが必ず 1 ビットだけ異なる」符号。QAM では
    隣接シンボルへの誤判定が最も起こりやすいので、隣接シンボルにグレイ符号を
    割り当てておけば 1 シンボル誤りがほぼ 1 ビット誤りで済み、BER を抑えられる。

    変換式 g = n XOR (n >> 1) (右シフトとの排他的論理和) で得られる。
    入力・出力とも同じ shape の整数配列 (要素ごとに変換)。
    """
    return n ^ (n >> 1)


def _gray_decode(g: np.ndarray, nbits: int) -> np.ndarray:
    """グレイ符号 -> 通常の2進整数 (_gray_encode の逆変換)。

    復号式は b = g ^ (g>>1) ^ (g>>2) ^ ... ^ (g>>(nbits-1))。
    エンコードが 1 段の右シフト XOR なのに対し、デコードは全段の右シフトを
    XOR で累積する必要がある (誤差の伝播を打ち消すため)。

    Args:
        g: グレイ符号の整数配列。
        nbits: 1 値あたりのビット数 (シフトを回す回数を決める)。

    Returns:
        g と同じ shape の通常 2 進整数配列 (= 振幅順のレベル番号)。
    """
    b = g.copy()
    for s in range(1, nbits):
        b ^= (g >> s)                        # 各段の右シフトを累積 XOR
    return b


def _int_to_bits(vals: np.ndarray, nbits: int) -> np.ndarray:
    """整数配列 -> (len, nbits) のビット行列 (MSB first)。

    各整数を nbits ビットの 2 進表現に展開する。MSB first なので 0 列目が
    最上位ビット。

    Args:
        vals: 整数配列 (shape=(len,))。各値は 0..2**nbits-1。
        nbits: 展開するビット数。

    Returns:
        ビット行列 (shape=(len, nbits), dtype=np.int8)。
    """
    # shifts = [nbits-1, ..., 1, 0]。各列を対応する桁だけ右シフトして最下位ビットを抽出
    shifts = np.arange(nbits - 1, -1, -1)
    # vals[:, None] で列ベクトル化し、shifts (行) とブロードキャストして一括展開
    return ((vals[:, None] >> shifts) & 1).astype(np.int8)


def _bits_to_int(bits: np.ndarray) -> np.ndarray:
    """(len, nbits) のビット行列 (MSB first) -> 整数配列 (_int_to_bits の逆)。

    Args:
        bits: ビット行列 (shape=(len, nbits))。0 列目が最上位ビット。

    Returns:
        整数配列 (shape=(len,), dtype=np.int64)。
    """
    nbits = bits.shape[1]
    # 各桁の重み [2**(nbits-1), ..., 2, 1] (MSB first に対応)
    weights = (1 << np.arange(nbits - 1, -1, -1))
    # 行列×重みベクトルの内積で、各行のビットを 1 つの整数にまとめる
    return bits.astype(np.int64) @ weights


# =============================================================================
# 方形QAM
# =============================================================================
def qam_params(M: int):
    """方形QAM の基本パラメータ (k, L, kk) を計算して返す。

    方形 M-QAM は I 軸 (実部) と Q 軸 (虚部) をそれぞれ √M 値の PAM とみなした
    格子状コンステレーション。ここでその構造を表す 3 つの量を求める。

    Args:
        M: QAM の多値数。方形 QAM (4, 16, 64, 256, ...) のみ許す。

    Returns:
        (k, L, kk) のタプル:
          k  = log2(M)  … 1 シンボルが運ぶビット数
          L  = √M       … 1 軸あたりの振幅レベル数
          kk = k // 2   … 1 軸あたりのビット数 (前半 I 軸 / 後半 Q 軸に等分)
    """
    k = int(round(np.log2(M)))
    L = int(round(np.sqrt(M)))
    # √M が整数で、かつ k が偶数 (= I/Q に等分できる) でなければ方形 QAM ではない
    if L * L != M or (k % 2) != 0:
        raise ValueError(f"M={M} は方形QAM (4,16,64,256,...) ではありません")
    return k, L, k // 2


def qam_norm(M: int) -> float:
    """平均シンボル電力を 1 に規格化するための割り算係数 sqrt(2(M-1)/3)。

    整数振幅 {±1, ±3, ..., ±(L-1)} の格子で QAM 点を作ると、平均電力は
    E[|s|^2] = 2(M-1)/3 になる。これを 1 にそろえる (システム間で SNR を
    公平に比較するため) には、振幅をその平方根 sqrt(2(M-1)/3) で割ればよい。

    Args:
        M: QAM 多値数。

    Returns:
        規格化係数 (スカラ float)。
    """
    return np.sqrt(2.0 * (M - 1) / 3.0)


def qam_constellation(M: int) -> np.ndarray:
    """規格化済みの全 M 個のコンステレーション点を返す (シンボル番号順ではない)。

    I 軸・Q 軸の振幅レベルの直積として、格子状に並ぶ全 M 点を生成する。
    主に図示や、平均電力が厳密に 1 であることの確認に使う。

    Args:
        M: QAM 多値数。

    Returns:
        複素コンステレーション点 (shape=(M,), dtype=complex)。平均 |点|^2 = 1。
    """
    _, L, _ = qam_params(M)
    levels = 2 * np.arange(L) - (L - 1)           # 1 軸の振幅 {-(L-1),...,-1,1,...,(L-1)}
    I, Q = np.meshgrid(levels, levels)            # I/Q の全組み合わせ (格子) を作る
    pts = (I.ravel() + 1j * Q.ravel()) / qam_norm(M)  # 複素点にして平均電力 1 へ規格化
    return pts


def bits_to_symbols(bits: np.ndarray, M: int) -> np.ndarray:
    """ビット列 -> 規格化複素QAMシンボル列 (QAMマッピング / 変調)。

    送信ビット列を QAM シンボルに変換する変調器。1 シンボル k=log2(M) ビットを
    取り出し、前半 kk ビットを I 軸、後半 kk ビットを Q 軸のグレイ符号として扱う。
    グレイ符号を通常整数 (振幅順レベル番号 0..L-1) に直してから、整数振幅
    2*idx-(L-1) に変換し、最後に平均電力 1 へ規格化する。

    Args:
        bits: 0/1 ビット列 (shape=(N*k,))。長さは log2(M) の倍数であること。
        M: QAM 多値数。

    Returns:
        複素 QAM シンボル列 (shape=(N,), dtype=complex)。平均電力 ≈ 1。

    第2週の QAM 変復調の問題で使用する中心関数。
    """
    k, L, kk = qam_params(M)
    bits = np.asarray(bits, dtype=np.int8)
    if bits.size % k != 0:
        raise ValueError(f"ビット数 {bits.size} が log2(M)={k} の倍数ではありません")
    g = bits.reshape(-1, k)                         # 1 行 = 1 シンボル分の k ビット

    i_gray = _bits_to_int(g[:, :kk])               # 前半 kk ビット = I軸グレイ符号 (整数)
    q_gray = _bits_to_int(g[:, kk:])               # 後半 kk ビット = Q軸グレイ符号 (整数)
    i_idx = _gray_decode(i_gray, kk)               # グレイ符号 -> 振幅順レベル番号 0..L-1
    q_idx = _gray_decode(q_gray, kk)

    # レベル番号 0..L-1 を等間隔の整数振幅 {-(L-1),...,(L-1)} へ写す
    i_amp = 2 * i_idx - (L - 1)
    q_amp = 2 * q_idx - (L - 1)
    return (i_amp + 1j * q_amp) / qam_norm(M)       # 複素化して平均電力 1 に規格化


def _slice_indices(sym: np.ndarray, M: int):
    """規格化シンボル -> 各軸の最近傍レベル番号 (0..L-1) を返す (硬判定の中核)。

    bits_to_symbols の逆計算。規格化を解いて整数振幅スケールに戻し、
    振幅 amp = 2*idx-(L-1) を idx について解くと idx = (amp+(L-1))/2。
    これを四捨五入 (rint) して最近傍レベルに丸め、はみ出しを clip で抑える。

    Args:
        sym: 受信複素シンボル (雑音を含んでよい)。
        M: QAM 多値数。

    Returns:
        (i_idx, q_idx): 各軸のレベル番号 0..L-1 の整数配列 (sym と同 shape)。
    """
    _, L, _ = qam_params(M)
    x = sym * qam_norm(M)                          # 規格化を解いて整数振幅スケールに戻す
    # 振幅 -> レベル番号に変換し、四捨五入で最近傍へ、clip で 0..L-1 に収める
    i_idx = np.clip(np.rint((x.real + (L - 1)) / 2.0), 0, L - 1).astype(np.int64)
    q_idx = np.clip(np.rint((x.imag + (L - 1)) / 2.0), 0, L - 1).astype(np.int64)
    return i_idx, q_idx


def decide(sym: np.ndarray, M: int) -> np.ndarray:
    """硬判定: 各サンプルを最も近いコンステレーション点へ丸める。

    雑音でずれた受信点を、最近傍の理想シンボル点に置き換える (判定)。
    出力は再び規格化済みシンボル。コンステレーション図の判定境界の確認や、
    判定後シンボルを使う処理に用いる。

    Args:
        sym: 受信複素シンボル列。
        M: QAM 多値数。

    Returns:
        最近傍に丸めた規格化シンボル列 (sym と同 shape, dtype=complex)。
    """
    _, L, _ = qam_params(M)
    i_idx, q_idx = _slice_indices(sym, M)         # 各軸の最近傍レベル番号
    i_amp = 2 * i_idx - (L - 1)                    # レベル番号 -> 整数振幅
    q_amp = 2 * q_idx - (L - 1)
    return (i_amp + 1j * q_amp) / qam_norm(M)      # 規格化して理想点へ戻す


def symbols_to_bits(sym: np.ndarray, M: int) -> np.ndarray:
    """シンボル列 -> ビット列 (硬判定 + QAMデマッピング / 復調)。

    bits_to_symbols の逆処理を行う復調器。入力が雑音を含む受信シンボルでも、
    内部で最近傍レベルに硬判定してから、レベル番号をグレイ符号化し直し、
    I 軸・Q 軸の順にビットへ展開して連結する。

    Args:
        sym: 受信複素シンボル列 (shape=(N,))。
        M: QAM 多値数。

    Returns:
        復調ビット列 (shape=(N*k,), dtype=np.int8)。bits_to_symbols と
        同じ並び (シンボルごとに I 軸 kk ビット + Q 軸 kk ビット)。
    """
    _, L, kk = qam_params(M)
    i_idx, q_idx = _slice_indices(sym, M)          # 最近傍レベル番号 0..L-1
    i_gray = _gray_encode(i_idx)                    # レベル番号 -> グレイ符号 (変調時と整合)
    q_gray = _gray_encode(q_idx)
    i_bits = _int_to_bits(i_gray, kk)              # 各軸グレイ符号をビット行列 (N, kk) に
    q_bits = _int_to_bits(q_gray, kk)
    # I軸ビット | Q軸ビット を列方向に連結し、行優先で平坦化して 1 本のビット列に
    return np.concatenate([i_bits, q_bits], axis=1).ravel()


# =============================================================================
# 雑音とBER
# =============================================================================
def add_awgn(sym: np.ndarray, snr_db: float, signal_power: float | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """複素AWGN (白色ガウス雑音) を SNR [dB] 指定で付加する。

    通信路の熱雑音などをモデル化した加法的白色ガウス雑音 (AWGN) を信号に加える。
    SNR = (信号電力) / (雑音電力)。dB を真数に直して目標雑音電力を決める:
        snr_lin = 10^(snr_db/10),  雑音電力 N0 = signal_power / snr_lin。
    複素雑音 n = n_re + j*n_im は実部・虚部が独立同分布なので、総電力 N0 を
    実部・虚部に半分ずつ配分する。よって 1 軸あたりの分散は N0/2、
    標準偏差は sigma = sqrt(N0/2) となる。

    Args:
        sym: 複素ベースバンド信号 (任意 shape)。
        snr_db: 信号電力対雑音電力比 [dB]。
        signal_power: 信号電力。省略時は sym から実測 (平均 |sym|^2)。
        rng: 乱数生成器 (再現性のため指定可)。省略時は新規生成。

    Returns:
        雑音付加後の複素信号 (sym と同 shape)。
    """
    if rng is None:
        rng = np.random.default_rng()
    if signal_power is None:
        signal_power = np.mean(np.abs(sym) ** 2)  # 信号電力 = 平均 |sym|^2
    snr_lin = 10 ** (snr_db / 10.0)               # dB -> 真数 (線形比)
    noise_power = signal_power / snr_lin           # 複素雑音の総電力 N0
    sigma = np.sqrt(noise_power / 2.0)             # 1軸 (実部 or 虚部) あたりの標準偏差
    # 独立な実部・虚部ガウス雑音を生成 (各分散 sigma^2 で合計が N0 になる)
    noise = sigma * (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape))
    return sym + noise


def count_bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> int:
    """送信/受信ビット列の不一致 (ビット誤り) 数を数える。

    BER (ビット誤り率) = この戻り値 / 比較ビット数 として使う。長さが違う
    場合は短い方に合わせて比較する。

    Args:
        tx_bits: 送信ビット列。
        rx_bits: 受信 (復調) ビット列。

    Returns:
        一致しなかったビットの個数 (int)。
    """
    n = min(len(tx_bits), len(rx_bits))            # 短い方の長さに合わせる
    return int(np.count_nonzero(tx_bits[:n] != rx_bits[:n]))  # 不一致 (True) の数を数える


def ber_theory_qam(M: int, snr_db, snr_per: str = "symbol") -> np.ndarray:
    """グレイ符号化 方形M-QAM の BER 解析解 (近似式) を返す。

    シミュレーションで測った BER を比較・検算するための理論曲線。グレイ符号
    方形 QAM の高 SNR 近似式:

        BER ≈ (4/k)(1 - 1/√M) Q( sqrt( 3/(M-1) · γ_s ) )

    各記号: k=log2(M) はシンボルあたりビット数、√M は軸あたりレベル数、
    γ_s は 1 シンボルあたりの SNR (Es/N0)、Q(·) は標準正規の上側確率
    Q(x) = (1/2) erfc(x/√2)。QPSK (M=4) では厳密解 Q(√γ_s) に一致する。

    Args:
        M: QAM多値数。
        snr_db: SNR [dB] (スカラまたは配列)。
        snr_per: SNR の基準。"symbol" なら γ_s = 10^(snr/10)、
            "bit" なら γ_b として Es/N0 = k·γ_b (シンボルあたりに換算)。

    Returns:
        各 SNR に対する BER (snr_db と同 shape の配列)。
    """
    from math import erfc  # スカラ用。配列は scipy 非依存で自前 Q を使う。

    k, L, _ = qam_params(M)
    snr_db = np.asarray(snr_db, dtype=float)
    snr_lin = 10 ** (snr_db / 10.0)               # dB -> 真数
    # "bit" 指定なら Eb/N0 をシンボルあたり Es/N0 = k·(Eb/N0) に換算
    gamma_s = snr_lin if snr_per == "symbol" else snr_lin * k

    # Q(x) = 0.5 * erfc(x/sqrt(2))。math.erfc はスカラ用なので vectorize で配列対応に
    from numpy import vectorize
    qfunc = vectorize(lambda x: 0.5 * erfc(x / np.sqrt(2.0)))
    arg = np.sqrt(3.0 / (M - 1) * gamma_s)        # Q 関数の引数 (隣接判定距離 / 雑音)
    return (4.0 / k) * (1.0 - 1.0 / L) * qfunc(arg)  # 先頭係数で 1 シンボル誤りをビット誤りに換算


# =============================================================================
# 2x2 MIMO LMS 適応等化 (バタフライ構成)
# =============================================================================
def mimo_lms(rx: np.ndarray, ref: np.ndarray, ntaps: int, mu: float):
    """偏波多重信号の 2x2 MIMO LMS 等化 (データ援用 / シンボル間隔)。

    偏波多重 (X 偏波・Y 偏波) では伝送路で 2 偏波が混じり合う。これを 4 本の
    FIR フィルタ (バタフライ構成) で分離・等化する。係数は LMS (最小平均二乗)
    アルゴリズムで、参照シンボルとの誤差を使って 1 サンプルずつ更新する
    (データ援用 = 既知の送信シンボルを教師に使う)。

    バタフライ FIR の出力 (j はタップ番号):
        v_x[n] = Σ_j W_xx[j] u_x[..] + W_xy[j] u_y[..]
        v_y[n] = Σ_j W_yx[j] u_x[..] + W_yy[j] u_y[..]
    誤差 e = d - v (d は参照シンボル) に対する LMS 更新式:
        W += mu * e * conj(u)
    conj(u) は複素 LMS の勾配方向、mu は収束速度と安定性を決めるステップサイズ。

    Args:
        rx:  受信両偏波信号 (shape=(2, N)、0 行目=X 偏波, 1 行目=Y 偏波)。
        ref: 参照(送信)両偏波シンボル (shape=(2, N))。
        ntaps: タップ数 (1 なら単一タップ = 2x2 行列のみの逆混合)。
        mu: ステップサイズ (収束速度。大きすぎると発散)。

    Returns:
        (v_out, valid): v_out は等化出力 (shape=(2, N))、valid は端の過渡を
        除いた有効サンプルのインデックス範囲 (slice)。中心タップ基準で
        参照を合わせているため、有効範囲は両端を c だけ内側に詰めた区間。
    """
    L = ntaps
    two, N = rx.shape
    c = L // 2                                 # 中心タップ位置 (群遅延の基準)
    # W[a,b] は「入力偏波 b から出力偏波 a」への長さ L の FIR 係数
    W = np.zeros((2, 2, L), dtype=complex)
    W[0, 0, c] = 1.0                           # 中心タップを単位行列で初期化 (恒等から学習開始)
    W[1, 1, c] = 1.0
    v_out = np.zeros((2, N), dtype=complex)

    # n は窓の右端 (最新サンプル)。窓が信号に収まる L-1 から開始
    for n in range(L - 1, N):
        ux = rx[0, n - L + 1:n + 1]            # X偏波の入力窓 長さ L (古い→新しい)
        uy = rx[1, n - L + 1:n + 1]            # Y偏波の入力窓
        # 各出力 = 自偏波 FIR + 他偏波 FIR (バタフライ加算)
        vx = np.dot(W[0, 0], ux) + np.dot(W[0, 1], uy)
        vy = np.dot(W[1, 0], ux) + np.dot(W[1, 1], uy)
        t = n - L + 1 + c                       # 中心タップが対応する時刻 (出力シンボル位置)
        dx = ref[0, t]                          # その時刻の参照 (教師) シンボル
        dy = ref[1, t]
        ex = dx - vx                            # X偏波の推定誤差
        ey = dy - vy                            # Y偏波の推定誤差
        # 複素 LMS 更新: 各タップを 誤差 × 入力共役 の方向へ mu だけ動かす
        W[0, 0] += mu * ex * np.conj(ux)
        W[0, 1] += mu * ex * np.conj(uy)
        W[1, 0] += mu * ey * np.conj(ux)
        W[1, 1] += mu * ey * np.conj(uy)
        v_out[0, t] = vx                        # 等化出力を中心タップ時刻に格納
        v_out[1, t] = vy

    # 先頭 c サンプルと末尾 (L-1-c) サンプルは窓が埋まらず未計算なので除外
    valid = slice(c, N - (L - 1 - c))
    return v_out, valid


def mimo_lms_fse(rx: np.ndarray, ref: np.ndarray, ntaps: int, mu: float,
                 sps: int, base: int):
    """分数間隔 (fractionally-spaced) 2x2 MIMO LMS 等化。

    mimo_lms のオーバーサンプリング版。入力を sps sample/symbol のまま受け、
    シンボルレートで 1 出力を出す (窓を sps サンプルずつスライド)。タップ間隔が
    シンボル周期より細かい (分数間隔) ため、整合フィルタ・タイミング補償・
    偏波分離をまとめて学習でき、サンプリング位相のずれに強い。更新式は
    mimo_lms と同じ複素 LMS (W += mu*e*conj(u))。

    出力シンボル k は入力窓 rx[:, k*sps+base : k*sps+base+ntaps] を使う。

    Args:
        rx:  受信両偏波信号 (shape=(2, Nsamp))。
        ref: 参照(送信)両偏波シンボル (shape=(2, Nsym))。
        ntaps: タップ数。
        mu: ステップサイズ。
        sps: 1シンボルあたりサンプル数 (= 窓のスライド量)。
        base: k=0 のときの入力開始インデックス (群遅延・タイミング位相を含む)。

    Returns:
        (v_out, kmax): v_out は等化出力 (shape=(2, Nsym))、kmax は有効に
        計算できた最大シンボル番号 (それ以降は窓が信号外なので未計算)。
    """
    L = ntaps
    Nsym = ref.shape[1]
    Nsamp = rx.shape[1]
    W = np.zeros((2, 2, L), dtype=complex)
    W[0, 0, L // 2] = 1.0                       # 中心タップを単位行列で初期化
    W[1, 1, L // 2] = 1.0
    v_out = np.zeros((2, Nsym), dtype=complex)
    # 窓 [k*sps+base, +L) が信号に収まる最大の k (かつ参照の範囲内)
    kmax = min((Nsamp - base - L) // sps, Nsym - 1)
    # 窓の左端 k*sps+base が 0 以上になる最小の k (base が負なら正へ繰り上げ)
    k0 = max(0, (-base + sps - 1) // sps)
    for k in range(k0, kmax + 1):
        i0 = k * sps + base                     # シンボル k に対応する入力窓の開始位置
        ux = rx[0, i0:i0 + L]                   # X偏波 入力窓 (長さ L、分数間隔)
        uy = rx[1, i0:i0 + L]                   # Y偏波 入力窓
        vx = np.dot(W[0, 0], ux) + np.dot(W[0, 1], uy)  # バタフライ出力 (X)
        vy = np.dot(W[1, 0], ux) + np.dot(W[1, 1], uy)  # バタフライ出力 (Y)
        ex = ref[0, k] - vx                     # X偏波 誤差 (参照 - 出力)
        ey = ref[1, k] - vy                     # Y偏波 誤差
        W[0, 0] += mu * ex * np.conj(ux)        # 複素 LMS 更新 (4 本のタップ)
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

    現実のレーザ光は位相が時間とともにふらつく (位相雑音)。これをウィーナー
    過程 (ランダムウォーク) でモデル化する。各ステップの位相増分が独立な
    ガウス雑音で、その分散はレーザ線幅 df とサンプル周期 1/fs に比例する:

        θ[0] = uniform(-π, π),   θ[n] = θ[n-1] + w[n],  w[n] ~ N(0, 2π·df/fs)

    線幅 df が広い (= レーザの質が低い) ほど位相が速くばらつく。コヒーレント
    受信での搬送波位相推定 (CPE) の演習で使う。

    Args:
        n:  サンプル数。
        df: スペクトル線幅 (FWHM) [Hz]。
        fs: サンプリングレート [Hz]。
        rng: 乱数生成器。省略時は新規生成。

    Returns:
        長さ n の位相系列 [rad] (shape=(n,))。
    """
    if rng is None:
        rng = np.random.default_rng()
    var = 2.0 * np.pi * df / fs                     # 1 ステップの位相増分の分散
    incr = rng.standard_normal(n) * np.sqrt(var)    # ガウス位相増分 w[n]
    incr[0] = rng.uniform(-np.pi, np.pi)            # 初期位相だけは -π〜π の一様分布で置換
    return np.cumsum(incr)                           # 累積和でランダムウォークにする


# =============================================================================
# レイズドコサイン (パルス整形) フィルタ
# =============================================================================
def rc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """レイズドコサイン (RC) フィルタのインパルス応答。

    符号間干渉 (ISI) を起こさないパルス整形フィルタ。インパルス応答は
        h(t) = sinc(t/T) · cos(π·β·t/T) / (1 - (2·β·t/T)^2)
    で、ナイキスト条件 (t = 整数倍シンボル時刻で 0、中心で 1) を満たす。
    β はロールオフ (帯域の広がり) を決める。ここでは t をシンボル時間単位
    (t/T) で扱う。

    Args:
        beta: ロールオフ係数 α (0〜1)。
        sps:  1シンボルあたりサンプル数 (oversampling factor)。
        span: フィルタ長 (シンボル数)。全長 = span*sps+1。

    Returns:
        ピークが 1 に規格化された対称 FIR 係数 (shape=(span*sps+1,))。
    """
    n = np.arange(-span * sps / 2, span * sps / 2 + 1)  # サンプル位置 (中心 0 の対称)
    t = n / sps                                    # シンボル時間単位 (t/T) に換算
    sinc = np.sinc(t)                              # ナイキストの sinc 項 (numpy は正規化 sinc)
    with np.errstate(divide="ignore", invalid="ignore"):
        # コサイン項。分母 1-(2βt)^2 が 0 になる点は一旦 inf/nan を許容して後で埋める
        cos = np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
    # 特異点 (1-(2βt)^2 = 0, すなわち t = ±1/(2β)) は 0/0 になるので極限値 π/4 で埋める
    if beta > 0:
        sing = np.isclose(np.abs(2 * beta * t), 1.0)
        cos[sing] = np.pi / 4
    h = sinc * cos
    return h


def pulse_shape(sym: np.ndarray, sps: int, beta: float, span: int):
    """シンボル列を raised cosine で整形する (0挿入アップサンプル + 畳み込み)。

    各シンボルの間に (sps-1) 個の 0 を挿入してサンプルレートを上げ (アップ
    サンプル)、RC フィルタと畳み込むことで帯域制限された連続波形に近づける。

    Args:
        sym: シンボル列 (shape=(Nsym,))。
        sps: 1シンボルあたりサンプル数。
        beta: ロールオフ係数。
        span: フィルタ長 (シンボル数)。

    Returns:
        (shaped, delay): shaped は整形サンプル列、delay は群遅延 [サンプル]。
        shaped[delay + n*sps] が n 番目のシンボル中心 (ISI なしのタイミング)。
    """
    h = rc_filter(beta, sps, span)
    up = np.zeros(len(sym) * sps, dtype=complex)   # アップサンプル用のゼロ配列
    up[::sps] = sym                                # sps 間隔にシンボルを配置 (間は 0)
    shaped = np.convolve(up, h)                    # フィルタと畳み込んで波形整形
    delay = (len(h) - 1) // 2                      # 対称 FIR の群遅延 (中心タップまでの長さ)
    return shaped, delay


def downsample_isi_free(shaped: np.ndarray, delay: int, sps: int, n_sym: int) -> np.ndarray:
    """整形サンプル列から、ISIが生じないシンボル中心タイミングを抽出する。

    pulse_shape の逆操作 (ダウンサンプル)。群遅延 delay を起点に sps 間隔で
    サンプルを拾うと、各シンボル中心 (他シンボルの裾が 0 になる点) が取れる。

    Args:
        shaped: 整形サンプル列。
        delay: pulse_shape が返した群遅延 [サンプル]。
        sps: 1シンボルあたりサンプル数。
        n_sym: 取り出すシンボル数。

    Returns:
        シンボル中心サンプル列 (長さ ≤ n_sym、末尾が範囲外なら切り詰め)。
    """
    idx = delay + np.arange(n_sym) * sps          # 各シンボル中心のサンプル位置
    idx = idx[idx < len(shaped)]                  # 配列範囲を超えるインデックスは除外
    return shaped[idx]


def rrc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """ルートレイズドコサイン (RRC) フィルタのインパルス応答 (エネルギー規格化)。

    RC を「平方根」に分けたフィルタ。送受信に同じ RRC を置くと、その縦続
    (RRC * RRC) が RC になり整合フィルタとしてナイキスト条件を満たす。
    閉形式に 0/0 となる 2 種類の特異点 (t=0 と t=±1/(4β)) があり、それぞれ
    極限値で個別に計算する。係数はエネルギー (Σh^2) が 1 になるよう規格化。

    Args:
        beta: ロールオフ係数 α (0〜1)。
        sps:  1シンボルあたりサンプル数。
        span: フィルタ長 (シンボル数)。

    Returns:
        エネルギー規格化された対称 FIR 係数 (shape=(span*sps+1,))。
    """
    N = span * sps
    n = np.arange(-N / 2, N / 2 + 1)              # サンプル位置 (中心 0 の対称)
    t = n / sps                                    # シンボル時間単位 (t/T)
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            # 特異点 t=0: 一般式の極限値
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4 * beta)):
            # 特異点 t=±1/(4β): 分母が 0 になるためロピタルの定理による極限値
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            # 通常点: RRC の閉形式 num/den
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))                  # エネルギーを 1 に規格化 (整合フィルタ用)
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

    # 2) QAM マッピングの往復 (無雑音なら完全復元) と平均電力 1 の確認
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
