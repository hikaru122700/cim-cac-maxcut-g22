"""光ファイバ伝搬ライブラリ (fiber.py)

第4週 (演習問題 20〜26) で使う「光ファイバ中をパルスがどう変化しながら伝わるか」を
シミュレーションするための部品集。光信号の複素振幅 A(z, T) を時間軸 T 上の配列として持ち、
距離 z 方向に少しずつ進めていく。

■ 単位系 — すべて [ps], [km], [W] 系で統一する。
  - 時間 t (= 遅延時間 T): ps  (= 10^-12 s)
  - 距離 z:                 km
  - 振幅 A:                 √W  (|A|^2 がそのまま瞬時パワー [W] になるよう規格化)
  - 二次分散 β2:            ps^2/km  (群速度分散 GVD。波長によりパルスが広がる効果)
  - 非線形係数 γ:           1/(W·km) (Kerr 効果の強さ。光強度で屈折率が変わる)
  - 損失 α:                 1/km     (振幅の減衰率。dB/km は alpha_from_db で換算)

■ 支配方程式 — 非線形シュレディンガー方程式 (NLSE, 損失込み):
    ∂A/∂z = -(α/2)A  - j(β2/2) ∂²A/∂T²  + jγ|A|²A
            ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^
            損失        波長分散 (GVD)         自己位相変調 (SPM)

  右辺の3項はそれぞれ「損失」「分散」「非線形(SPM)」に対応する物理効果。
  これらは数学的には同時に作用するが、解析解が無いので数値的に解く必要がある。

■ 解法 — Split-Step Fourier 法 (SSFM):
  微小区間 dz では「分散だけ作用」「非線形だけ作用」を交互に近似適用してよい、という考え方。
  分散項は ∂²/∂T² を含むので周波数領域 (FFT 後) で、非線形項は |A|² を含むので時間領域で
  適用すると、それぞれ単なる位相回転 (掛け算) になり計算が簡単になる。propagate_ssfm を参照。
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# パルス生成
# =============================================================================
def t0_from_fwhm_gauss(fwhm: float) -> float:
    """ガウスパルスの「強度 FWHM」から特性幅パラメータ T0 を求める。

    パルスは A(T) ∝ exp(-T²/(2 T0²)) と書ける。強度 |A|² ∝ exp(-T²/T0²) の
    半値全幅 (FWHM) は 2·√(ln2)·T0 になるので、これを逆に解いて T0 を得る。
    波形を作るときは「目で測れる FWHM」を入力に、式で使いやすい T0 へ変換する役割。

    Args:
        fwhm: 強度波形の半値全幅 [ps]。
    Returns:
        T0 [ps] (ガウスの 1/e 振幅半値幅に相当する特性幅)。
    """
    return fwhm / (2.0 * np.sqrt(np.log(2.0)))


def t0_from_fwhm_sech(fwhm: float) -> float:
    """sech パルスの「強度 FWHM」から特性幅 T0 を求める。

    A(T) ∝ sech(T/T0) のとき強度 |A|² = sech²(T/T0) の半値全幅は
    2·arccosh(√2)·T0 (≈ 1.7627·T0)。これを逆算して T0 を返す。
    sech 形はソリトン (問題22) の基本波形なのでこの換算が必要。

    Args:
        fwhm: 強度波形の半値全幅 [ps]。
    Returns:
        T0 [ps]。
    """
    return fwhm / (2.0 * np.arccosh(np.sqrt(2.0)))


def gaussian_pulse(t: np.ndarray, p_peak: float, fwhm: float) -> np.ndarray:
    """ガウス形の初期パルス振幅 A(T) を生成する (問題20〜21 などの入力波形)。

    A(T) = √P_peak · exp(-T² / (2 T0²)) を返す。√P_peak を掛けるのは
    |A|² がそのまま瞬時パワー [W] になるようにするため (ピークで |A|²=P_peak)。

    Args:
        t:      時間軸の配列 [ps], shape=(N,)。
        p_peak: 強度ピーク値 [W]。
        fwhm:   強度 FWHM [ps]。
    Returns:
        複素(ここでは実)振幅 A(T) の配列 [√W], shape=(N,)。
    """
    t0 = t0_from_fwhm_gauss(fwhm)               # FWHM → 特性幅 T0 へ変換
    return np.sqrt(p_peak) * np.exp(-t ** 2 / (2.0 * t0 ** 2))


def sech_pulse(t: np.ndarray, p_peak: float, fwhm: float) -> np.ndarray:
    """sech 形の初期パルス振幅 A(T) を生成する (ソリトン問題22 の入力波形)。

    A(T) = √P_peak / cosh(T/T0) = √P_peak · sech(T/T0)。
    sech 波形は分散と非線形が釣り合うとき形を保って伝わる (ソリトン) ため重要。

    Args:
        t:      時間軸の配列 [ps], shape=(N,)。
        p_peak: 強度ピーク値 [W]。
        fwhm:   強度 FWHM [ps]。
    Returns:
        振幅 A(T) の配列 [√W], shape=(N,)。
    """
    t0 = t0_from_fwhm_sech(fwhm)                # FWHM → 特性幅 T0 へ変換
    return np.sqrt(p_peak) / np.cosh(t / t0)


# =============================================================================
# パルス幅の測定
# =============================================================================
def fwhm_of(t: np.ndarray, power: np.ndarray) -> float:
    """強度波形の半値全幅 (FWHM) を線形補間で測定する。

    伝搬後にパルスがどれだけ広がったか (分散の効果) を数値で評価するための関数。
    ピーク強度の半分 (half = peak/2) を横切る左右2点を見つけ、サンプル間隔より
    細かい精度を得るため、その手前のサンプルとの間で線形補間して半値交差時刻を求める。

    Args:
        t:     時間軸の配列 [ps], shape=(N,)。等間隔・昇順を仮定。
        power: 強度波形 |A|² [W], shape=(N,)。
    Returns:
        FWHM (t と同じ単位 [ps])。半値以上のサンプルが1点以下なら測定不能で 0.0。
    """
    peak = power.max()
    half = peak / 2.0
    above = power >= half                       # 半値以上のサンプルを真偽マスクに
    idx = np.where(above)[0]                     # 半値以上になっているインデックス列
    if len(idx) < 2:
        return 0.0                              # 立ち上がり/立ち下がりが取れない → 測定不能
    iL, iR = idx[0], idx[-1]                     # 半値以上区間の左端・右端インデックス
    # 左端の半値交差: iL-1 (半値未満) と iL (半値以上) の2点を結ぶ直線で half を横切る時刻を補間。
    # np.interp は x が昇順であることを要求するため、x に power 値・y に t を渡し
    # 区間内で power が単調になっている前提で交差点 tL を求める。
    if iL > 0:
        tL = np.interp(half, [power[iL - 1], power[iL]], [t[iL - 1], t[iL]])
    else:
        tL = t[iL]                              # 端に張り付いている場合は補間できずサンプル時刻で代用
    # 右端の半値交差: iR (半値以上) と iR+1 (半値未満) の2点で補間。
    # 右側は power が減少するので、interp の x が昇順になるよう [iR+1, iR] の順で渡している。
    if iR < len(t) - 1:
        tR = np.interp(half, [power[iR + 1], power[iR]], [t[iR + 1], t[iR]])
    else:
        tR = t[iR]
    return abs(tR - tL)                          # 左右交差時刻の差 = 半値全幅


# =============================================================================
# 伝搬の各ステップ
# =============================================================================
def omega_grid(n: int, dt: float) -> np.ndarray:
    """FFT 用の角周波数グリッド ω [rad/ps] を作る。

    分散ステップを周波数領域で計算するため、各 FFT 成分に対応する角周波数が必要。
    np.fft.fftfreq は [0, +, ..., -, ...] という FFT 特有の並び (折り返し順) で
    周波数 [1/ps] を返すので、角周波数にするため 2π を掛ける。

    Args:
        n:  サンプル数 (= len(A))。
        dt: サンプリング間隔 [ps]。
    Returns:
        角周波数配列 ω [rad/ps], shape=(n,)。並びは fft の出力順に対応。
    """
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dt)


def dispersion_step(A: np.ndarray, beta2: float, dz: float, omega: np.ndarray) -> np.ndarray:
    """波長分散 (GVD) を距離 dz だけ作用させる (周波数領域での2次位相回転)。

    NLSE の分散項 -j(β2/2) ∂²A/∂T² は、FFT で時間→周波数に移すと ∂/∂T → jω に置き換わり
    単なる掛け算 exp(j (β2/2) ω² dz) になる。これを使って:
        A(z+dz) = IFFT[ FFT[A(z)] · exp(j (β2/2) ω² dz) ]
    各周波数成分に「ω に応じた位相遅れ」を与えることで、速い/遅い成分が分かれてパルスが広がる。
    位相は ω² (偶関数) なので FFT の符号規約に依らず正しく計算できる。

    Args:
        A:     入力振幅 A(T) [√W], shape=(N,)。
        beta2: 二次分散 β2 [ps²/km]。
        dz:    伝搬距離 [km]。
        omega: omega_grid で作った角周波数 [rad/ps], shape=(N,)。
    Returns:
        分散を受けた後の振幅 [√W], shape=(N,)。
    """
    # FFT → 各成分に2次位相 exp(j β2/2 · ω² · dz) を掛ける → IFFT で時間領域に戻す
    return np.fft.ifft(np.fft.fft(A) * np.exp(1j * (beta2 / 2.0) * omega ** 2 * dz))


def nonlinear_step(A: np.ndarray, gamma: float, dz: float) -> np.ndarray:
    """自己位相変調 (SPM) を距離 dz だけ作用させる (時間領域の位相回転)。

    NLSE の非線形項 jγ|A|²A は、振幅 |A| を変えずに位相だけを回す効果として表せる:
        A(z+dz) = A(z) · exp(j γ |A|² dz)
    強い (|A|² が大きい) 時刻ほど位相が大きく回るため、パルス内に瞬時周波数の偏り
    (チャープ) が生まれる。これが SPM によるスペクトル広がりの正体。

    Args:
        A:     入力振幅 A(T) [√W], shape=(N,)。
        gamma: 非線形係数 γ [1/(W·km)]。
        dz:    伝搬距離 [km]。
    Returns:
        SPM を受けた後の振幅 [√W], shape=(N,)。
    """
    # |A|² (= 瞬時パワー) に比例した位相を各時刻に与える。振幅の大きさは不変。
    return A * np.exp(1j * gamma * np.abs(A) ** 2 * dz)


def loss_step(A: np.ndarray, alpha: float, dz: float) -> np.ndarray:
    """ファイバ損失を距離 dz だけ作用させる (振幅の指数減衰)。

    NLSE の損失項 -(α/2)A はパワー P=|A|² が exp(-α dz) で減衰することを意味する。
    振幅 A は exp(-α/2 · dz) で減衰する (2乗してパワーの減衰率になる)。

    Args:
        A:     入力振幅 [√W], shape=(N,)。
        alpha: 損失係数 α [1/km] (振幅基準)。dB/km からは alpha_from_db で換算。
        dz:    伝搬距離 [km]。
    Returns:
        減衰後の振幅 [√W], shape=(N,)。
    """
    return A * np.exp(-alpha / 2.0 * dz)


def alpha_from_db(alpha_db_per_km: float) -> float:
    """損失係数を [dB/km] から振幅減衰係数 α [1/km] へ換算する。

    パワー基準で 10·log10(P_in/P_out) = α_dB·z [dB] という定義から、
    パワー減衰 exp(-α z) と突き合わせると α = α_dB / (10·log10 e) [1/km] となる。
    つまり α_dB ≈ 4.343·α。ファイバの仕様 (例: 0.2 dB/km) を計算用の α へ変換する。

    Args:
        alpha_db_per_km: 損失 [dB/km]。
    Returns:
        振幅減衰係数 α [1/km] (loss_step / propagate_ssfm にそのまま渡せる)。
    """
    return alpha_db_per_km / (10.0 * np.log10(np.e))


# =============================================================================
# Split-Step Fourier 法
# =============================================================================
def propagate_ssfm(A0: np.ndarray, z: float, dz: float, dt: float,
                   beta2: float = 0.0, gamma: float = 0.0, alpha: float = 0.0
                   ) -> np.ndarray:
    """対称 Split-Step Fourier 法 (SSFM) でパルスを距離 z だけ伝搬させる (問題20〜22 の本体)。

    分散・非線形・損失は同時に作用するが、微小区間 dz では交互適用で近似できる。
    本実装は誤差を小さくする「対称分割 (symmetric/Strang split)」を採用し、各 dz を
        半ステップ分散 (dz/2) → 非線形 + 損失 (dz 全部) → 半ステップ分散 (dz/2)
    の順に処理する。非線形ステップを2つの半分散で挟むことで、単純分割 O(dz) に対して
    1ステップあたりの分割誤差が O(dz²) に下がる (= より少ないステップで高精度)。

    beta2/gamma/alpha のいずれかを 0 にすれば、その物理効果だけを無効化して観察できる
    (例: gamma=0 で純粋な分散広がり、beta2=0 で純粋な SPM)。

    Args:
        A0:    初期振幅 A(z=0, T) [√W], shape=(N,)。
        z:     総伝搬距離 [km]。
        dz:    1ステップの目安距離 [km] (実際は z を割り切る値に丸め直す)。
        dt:    サンプリング間隔 [ps] (分散の周波数グリッド計算に使う)。
        beta2: 二次分散 β2 [ps²/km]。
        gamma: 非線形係数 γ [1/(W·km)]。
        alpha: 損失係数 α [1/km] (0 なら損失ステップを省略)。
    Returns:
        伝搬後の振幅 A(z, T) [√W], shape=(N,)。
    """
    A = A0.astype(complex).copy()               # 入力を壊さないよう複素配列にコピー
    omega = omega_grid(len(A), dt)              # 分散ステップ用の角周波数グリッド [rad/ps]
    nstep = max(1, int(round(z / dz)))          # ステップ数 (最低1。z/dz を四捨五入)
    dz = z / nstep                              # z を nstep で割り切れるよう dz を再定義
    # 半ステップ分散の演算子はループ中で不変なので、ループ外で一度だけ用意して使い回す。
    half_disp = np.exp(1j * (beta2 / 2.0) * omega ** 2 * (dz / 2.0))
    for _ in range(nstep):
        A = np.fft.ifft(np.fft.fft(A) * half_disp)          # 前半: 半ステップ分散 (dz/2)
        A = A * np.exp(1j * gamma * np.abs(A) ** 2 * dz)    # 中央: 非線形 SPM (dz 全部)
        if alpha:
            A = A * np.exp(-alpha / 2.0 * dz)               # 損失 (alpha=0 なら skip して高速化)
        A = np.fft.ifft(np.fft.fft(A) * half_disp)          # 後半: 半ステップ分散 (dz/2)
    return A


def dispersion_length(t0: float, beta2: float) -> float:
    """分散長 L_D = T0² / |β2| [km] を求める。

    分散だけでパルス幅が目立って広がり始める目安距離。伝搬距離 z をこの L_D と
    比べると分散効果の強さが見積もれる (z ≪ L_D ならほぼ広がらない)。

    Args:
        t0:    パルス特性幅 T0 [ps]。
        beta2: 二次分散 β2 [ps²/km]。
    Returns:
        分散長 L_D [km]。
    """
    return t0 ** 2 / abs(beta2)


# =============================================================================
# EDFA (光増幅器) と ASE 雑音
# =============================================================================
PLANCK = 6.62607015e-34          # プランク定数 [J·s]
LIGHT = 2.99792458e8             # 光速 [m/s]


def photon_energy(wavelength_nm: float = 1550.0) -> float:
    """光子1個のエネルギー hν [J] を求める (既定: 通信波長 1550 nm)。

    ν = c/λ (光速 ÷ 波長) で光周波数を出し、プランクの式 E = hν を計算する。
    ASE 雑音は「光子1個ぶん」を単位とするので、雑音電力の計算で基準量として使う。

    Args:
        wavelength_nm: 波長 [nm]。
    Returns:
        光子エネルギー hν [J]。
    """
    nu = LIGHT / (wavelength_nm * 1e-9)         # 波長[nm]→[m] に直して光周波数 ν=c/λ [Hz]
    return PLANCK * nu


def ase_psd(gain: float, nf_db: float, wavelength_nm: float = 1550.0) -> float:
    """EDFA 出力での ASE 雑音のパワースペクトル密度 (1偏波・片側) [W/Hz] を求める (問題25〜26)。

    光増幅器 (EDFA) は信号を増幅すると同時に自然放出由来の ASE 雑音を必ず加える。
    その密度は次式で与えられる:
        S_ASE = n_sp · hν · (G-1)
    ここで G は線形利得、hν は光子エネルギー、n_sp は自然放出係数。雑音指数 NF (dB) との
    関係は高利得近似で NF_lin = 2·n_sp なので n_sp = NF_lin/2 を使う。(G-1) は
    「増幅された自然放出ぶん」を表す因子。

    Args:
        gain:          線形利得 G (倍率。dB ではない)。
        nf_db:         雑音指数 NF [dB]。
        wavelength_nm: 波長 [nm]。
    Returns:
        片側 PSD S_ASE [W/Hz] (1偏波あたり)。
    """
    nf_lin = 10 ** (nf_db / 10.0)               # 雑音指数を dB → 線形倍率へ
    n_sp = nf_lin / 2.0                          # 自然放出係数 (高利得近似 NF_lin = 2 n_sp)
    return n_sp * photon_energy(wavelength_nm) * (gain - 1.0)


def edfa(A: np.ndarray, gain: float, nf_db: float, fs: float,
         rng: np.random.Generator, wavelength_nm: float = 1550.0) -> np.ndarray:
    """EDFA をモデル化する: 振幅を √G 倍し、ASE 雑音 (複素ガウス雑音) を加える (問題25〜26)。

    パワーが G 倍 = 振幅は √G 倍。これに加え、帯域 fs ぶんの ASE 雑音を複素 AWGN として乗せる。
    PSD S_ASE [W/Hz] に観測帯域 (= サンプリングレート fs [Hz]) を掛けると、
    1サンプルあたりの複素雑音の総電力 (分散) var = S_ASE·fs になる。複素雑音は実部・虚部の
    2軸に等分されるので、各軸の標準偏差は σ = √(var/2)。

    A の単位は √W (|A|² = 光パワー [W])。

    Args:
        A:             入力振幅 [√W], shape=(N,)。
        gain:          線形利得 G (倍率)。
        nf_db:         雑音指数 NF [dB]。
        fs:            サンプリングレート [Hz] (= 雑音を載せる帯域幅)。
        rng:           乱数生成器 (再現性のため呼び出し側で seed 管理)。
        wavelength_nm: 波長 [nm]。
    Returns:
        増幅 + 雑音付加後の振幅 [√W], shape=(N,)。
    """
    s_ase = ase_psd(gain, nf_db, wavelength_nm)     # ASE の PSD [W/Hz]
    var = s_ase * fs                                # 帯域 fs ぶんの複素雑音の総分散 [W]
    sigma = np.sqrt(var / 2.0)                      # 実部・虚部それぞれ (1軸) の標準偏差
    # 各サンプルに独立な複素ガウス雑音 (実部 + j·虚部) を生成
    noise = sigma * (rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape))
    return np.sqrt(gain) * A + noise                # 信号を √G 倍して ASE 雑音を加算
