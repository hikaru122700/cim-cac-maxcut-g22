# 問題4-4: EDFA 前置増幅構成での QPSK の BER vs 入力パワー Pin

## 問題

単一偏波 QPSK 光信号を EDFA で増幅した後、コヒーレント光受信する(**前置増幅構成**)。
EDFA から生じる **ASE 雑音は複素 AWGN** でモデル化でき、**雑音指数 NF = 4 dB** とする。
このときの **BER の EDFA 入力パワー $P_\mathrm{in}$ 依存性**を求める(レーザ位相雑音は無視)。

## 実行結果(出力)

```
Rs=32 Gbaud, NF=4.0 dB, G=20.0 dB
hν = 1.282e-19 J, S_ASE = 1.593e-17 W/Hz

 Pin[dBm]  Es/N0[dB]    BER(sim)  BER(theory)
      -50       2.92    8.04e-02     8.07e-02
      -45       7.92    6.54e-03     6.38e-03
      -43       9.92    8.43e-04     8.59e-04
      -40      12.92    3.75e-06     4.75e-06
      -38      14.92    0.00e+00     1.24e-08
受信感度 (BER=1e-3) ≈ -43.1 dBm
```

![前置増幅 QPSK の BER vs Pin](edfa_preamp_ber.png)

> **図の読み方** — 入力パワー $P_\mathrm{in}$ を上げるほど BER が急激に下がる。○(数値)が実線(理論)に一致。
> BER $=10^{-3}$ を切る入力パワー(**受信感度**)は約 $-43$ dBm で、前置増幅コヒーレント受信機の典型値。

---

## 1. 前置増幅構成と ASE 雑音

微弱な受信光をそのまま検出すると SNR が足りない。そこで受信機の前に **EDFA(エルビウム添加光ファイバ増幅器)**
を置いて増幅する(**前置増幅, preamplifier**)。ただし EDFA は信号を増幅すると同時に、自然放出光由来の
**ASE 雑音(Amplified Spontaneous Emission)** を加える。これがこの構成での主要な雑音源。

ASE 雑音は広帯域で各周波数独立、ガウス分布 → **複素 AWGN** でモデル化できる。
EDFA 出力での ASE のパワースペクトル密度(1偏波・片側)は

$$ S_\mathrm{ASE} = n_\mathrm{sp}\,h\nu\,(G-1),\qquad n_\mathrm{sp} = \frac{\mathrm{NF_{lin}}}{2}, $$

$h\nu$ は光子エネルギー($1550$ nm で $1.28\times10^{-19}$ J)、$G$ は利得、$n_\mathrm{sp}$ は自然放出係数。
**雑音指数 NF** は $\mathrm{NF_{lin}}=2n_\mathrm{sp}$(高利得近似)で結びつく。NF=4 dB → $\mathrm{NF_{lin}}=2.51$。

## 2. SNR と Pin の関係

1 sample/symbol($f_s=R_s$)で考えると、1シンボルあたりの ASE 雑音電力は $S_\mathrm{ASE}\cdot R_s$、
増幅後の信号電力は $G\,P_\mathrm{in}$。よって **1シンボル SNR(= $E_s/N_0$)** は

$$ \frac{E_s}{N_0} = \frac{G\,P_\mathrm{in}}{S_\mathrm{ASE}\,R_s}
= \frac{G\,P_\mathrm{in}}{n_\mathrm{sp}h\nu(G-1)R_s}
\;\xrightarrow[G\gg1]{}\; \frac{P_\mathrm{in}}{n_\mathrm{sp}h\nu R_s}
= \frac{2P_\mathrm{in}}{\mathrm{NF_{lin}}\,h\nu\,R_s}. $$

**SNR は $P_\mathrm{in}$ に比例**(利得 $G$ には実質よらない=増幅しても信号も雑音も同じだけ増えるから)。
QPSK の BER は

$$ \mathrm{BER} = Q\!\left(\sqrt{E_s/N_0}\right). $$

## 3. シミュレーションと一致

各 $P_\mathrm{in}$ について、QPSK 光($\sqrt{P_\mathrm{in}}\,s$)を EDFA に通して(`fiber.edfa`:$\sqrt G$ 倍 + ASE)、
正規化して QPSK 判定し BER を測る。出力のとおり、数値は理論とよく一致する。

**受信感度**(BER $=10^{-3}$ を達成する最小 $P_\mathrm{in}$)は約 $-43$ dBm。これは「1シンボルあたり
おおよそ $n_\mathrm{sp}\times$ 十数個の光子があれば誤りなく受信できる」という量子限界に近い感度で、
コヒーレント前置増幅受信機の高感度性を示している。

> **位相雑音を無視する意味** — 本問はレーザ位相雑音を無視するので、劣化要因は ASE のみ。
> 実機では問3 の位相回復 DSP で位相雑音を別途補償するため、ASE 限界の性能を議論できる。

## 4. コードのポイント

- `fiber.ase_psd(G, NF)` / `fiber.edfa(A, G, NF, fs, rng)` — ASE PSD と、増幅+ASE 付加。
- 信号は単位電力 QPSK に $\sqrt{P_\mathrm{in}}$ を掛けて光パワー $P_\mathrm{in}$ を表す。
- 受信後は $\sqrt{G P_\mathrm{in}}$ で正規化(QPSK 判定はスケールに依らないが、明示のため)。
- BER は対数軸。受信感度は BER 曲線を補間して求めている。

### 実行

```bash
python solution.py
```

## 5. この問題のつながり

ここでは EDFA 1台(前置増幅)。次の **問4-5** では、ファイバ損失 + EDFA を1スパンとして
**多中継伝送**(1〜50スパン)し、スパン数が増えると ASE が累積して BER が劣化する様子を見る。
**問4-6** で波長分散、**問4-7** で非線形効果を加えていく。
