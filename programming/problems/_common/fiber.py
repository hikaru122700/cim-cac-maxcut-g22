"""光ファイバ伝搬ライブラリ (fiber.py)

第4週で使う光ファイバ伝搬の部品。単位は [ps], [km], [W] 系で統一する。
  - 時間 t: ps
  - 距離 z: km
  - 二次分散 β2: ps^2/km
  - 非線形係数 γ: 1/(W·km)
  - 損失 α: 1/km (= dB/km を 10log10e で換算)

非線形シュレディンガー方程式 (NLSE, 損失込み):
    ∂A/∂z = -(α/2)A - j(β2/2) ∂²A/∂T² + jγ|A|²A

これを Split-Step Fourier 法で解く。分散項は周波数領域、非線形項は時間領域で適用する。
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# パルス生成
# =============================================================================
def t0_from_fwhm_gauss(fwhm: float) -> float:
    """ガウスパルスの強度 FWHM から 1/e 半値幅 T0 を求める。"""
    return fwhm / (2.0 * np.sqrt(np.log(2.0)))


def t0_from_fwhm_sech(fwhm: float) -> float:
    """sech パルスの強度 FWHM から T0 を求める。 |sech(T/T0)|^2 の FWHM = 2*arccosh(√2)*T0。"""
    return fwhm / (2.0 * np.arccosh(np.sqrt(2.0)))


def gaussian_pulse(t: np.ndarray, p_peak: float, fwhm: float) -> np.ndarray:
    """強度ピーク p_peak [W]、強度 FWHM [ps] のガウスパルス振幅 A(T)。"""
    t0 = t0_from_fwhm_gauss(fwhm)
    return np.sqrt(p_peak) * np.exp(-t ** 2 / (2.0 * t0 ** 2))


def sech_pulse(t: np.ndarray, p_peak: float, fwhm: float) -> np.ndarray:
    """強度ピーク p_peak [W]、強度 FWHM [ps] の sech パルス振幅 A(T)。"""
    t0 = t0_from_fwhm_sech(fwhm)
    return np.sqrt(p_peak) / np.cosh(t / t0)


# =============================================================================
# パルス幅の測定
# =============================================================================
def fwhm_of(t: np.ndarray, power: np.ndarray) -> float:
    """強度波形の半値全幅 (FWHM) を線形補間で求める [t と同じ単位]。"""
    peak = power.max()
    half = peak / 2.0
    above = power >= half
    idx = np.where(above)[0]
    if len(idx) < 2:
        return 0.0
    iL, iR = idx[0], idx[-1]
    # 左端: iL-1 -> iL を補間
    if iL > 0:
        tL = np.interp(half, [power[iL - 1], power[iL]], [t[iL - 1], t[iL]])
    else:
        tL = t[iL]
    if iR < len(t) - 1:
        tR = np.interp(half, [power[iR + 1], power[iR]], [t[iR + 1], t[iR]])
    else:
        tR = t[iR]
    return abs(tR - tL)


# =============================================================================
# 伝搬の各ステップ
# =============================================================================
def omega_grid(n: int, dt: float) -> np.ndarray:
    """角周波数グリッド [rad/ps] (fft 順序)。"""
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dt)


def dispersion_step(A: np.ndarray, beta2: float, dz: float, omega: np.ndarray) -> np.ndarray:
    """波長分散を距離 dz だけ適用する (周波数領域の2次位相)。

    分散演算子 exp(j (β2/2) ω² dz)。ω² は偶関数なので FFT の符号規約に依らず正しい。
    """
    return np.fft.ifft(np.fft.fft(A) * np.exp(1j * (beta2 / 2.0) * omega ** 2 * dz))


def nonlinear_step(A: np.ndarray, gamma: float, dz: float) -> np.ndarray:
    """自己位相変調 (SPM) を距離 dz だけ適用する (時間領域)。 exp(j γ |A|² dz)。"""
    return A * np.exp(1j * gamma * np.abs(A) ** 2 * dz)


def loss_step(A: np.ndarray, alpha: float, dz: float) -> np.ndarray:
    """損失を距離 dz だけ適用する。振幅は exp(-α/2 · dz)。"""
    return A * np.exp(-alpha / 2.0 * dz)


def alpha_from_db(alpha_db_per_km: float) -> float:
    """損失係数 [dB/km] を振幅減衰係数 α [1/km] に換算する。"""
    return alpha_db_per_km / (10.0 * np.log10(np.e))


# =============================================================================
# Split-Step Fourier 法
# =============================================================================
def propagate_ssfm(A0: np.ndarray, z: float, dz: float, dt: float,
                   beta2: float = 0.0, gamma: float = 0.0, alpha: float = 0.0
                   ) -> np.ndarray:
    """対称 Split-Step Fourier 法で距離 z だけ伝搬させる。

    各ステップ: 半分の分散 → 全部の非線形(+損失) → 半分の分散。
    beta2/gamma/alpha のいずれかを 0 にすれば、その効果だけ無効化できる。
    """
    A = A0.astype(complex).copy()
    omega = omega_grid(len(A), dt)
    nstep = max(1, int(round(z / dz)))
    dz = z / nstep
    half_disp = np.exp(1j * (beta2 / 2.0) * omega ** 2 * (dz / 2.0))
    for _ in range(nstep):
        A = np.fft.ifft(np.fft.fft(A) * half_disp)          # 半ステップ分散
        A = A * np.exp(1j * gamma * np.abs(A) ** 2 * dz)    # 非線形
        if alpha:
            A = A * np.exp(-alpha / 2.0 * dz)               # 損失
        A = np.fft.ifft(np.fft.fft(A) * half_disp)          # 半ステップ分散
    return A


def dispersion_length(t0: float, beta2: float) -> float:
    """分散長 L_D = T0² / |β2| [km]。"""
    return t0 ** 2 / abs(beta2)


# =============================================================================
# EDFA (光増幅器) と ASE 雑音
# =============================================================================
PLANCK = 6.62607015e-34          # プランク定数 [J·s]
LIGHT = 2.99792458e8             # 光速 [m/s]


def photon_energy(wavelength_nm: float = 1550.0) -> float:
    """光子エネルギー hν [J] (既定 1550 nm)。"""
    nu = LIGHT / (wavelength_nm * 1e-9)
    return PLANCK * nu


def ase_psd(gain: float, nf_db: float, wavelength_nm: float = 1550.0) -> float:
    """EDFA出力での ASE 雑音パワースペクトル密度 (1偏波・片側) [W/Hz]。

        S_ASE = n_sp · hν · (G-1),   n_sp = NF_lin / 2  (高利得近似 NF_lin = 2 n_sp)
    """
    nf_lin = 10 ** (nf_db / 10.0)
    n_sp = nf_lin / 2.0
    return n_sp * photon_energy(wavelength_nm) * (gain - 1.0)


def edfa(A: np.ndarray, gain: float, nf_db: float, fs: float,
         rng: np.random.Generator, wavelength_nm: float = 1550.0) -> np.ndarray:
    """EDFA: 振幅を √G 倍し、ASE 雑音 (複素AWGN) を付加する。

    A の単位は √W (|A|^2 = 光パワー [W])。fs はサンプリングレート [Hz]。
    ASE の1サンプルあたり複素分散 = S_ASE · fs (帯域 fs ぶんの雑音電力)。
    """
    s_ase = ase_psd(gain, nf_db, wavelength_nm)
    var = s_ase * fs                                # 複素雑音の総分散
    sigma = np.sqrt(var / 2.0)                      # 1軸あたり
    noise = sigma * (rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape))
    return np.sqrt(gain) * A + noise
