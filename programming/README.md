# プログラミング演習問題

光ディジタルコヒーレント通信の Python 演習問題集(全4週・26問)。
問題ごとに `problems/` 配下のフォルダに、解答コード `solution.py` と解説 `explanation.md` をまとめています。

入力課題(原本)は `input/` の docx を参照。各問は前の問の上に積み上がる構成で、
**第1週: QAM 送受信の基礎 → 第2週: BER 曲線とパルス整形 → 第3週: 位相雑音・偏波・MIMO →
第4週: 光ファイバ伝搬**、と進みます。

## ディレクトリ構成

```
programming/
├─ README.md                 # このファイル(問題一覧の索引)
├─ input/                    # 課題原本 (docx)
└─ problems/
   ├─ _common/               # 共通ライブラリ
   │  ├─ comm.py             #   通信系: PRBS, QAM, AWGN, BER, RC, 位相雑音, MIMO
   │  └─ fiber.py            #   光ファイバ系: 分散, 非線形, ソリトン, EDFA/ASE, Split-Step
   └─ NN_問題名/
      ├─ solution.py         # 解答コード
      └─ explanation.md      # 解説
```

## 共通ライブラリ

ほぼ全問が `_common/comm.py` と `_common/fiber.py` を import して使います。各 `solution.py` は
`sys.path` に `_common` を追加して読み込むので、各フォルダ単体で実行できます。

- `comm.py` — `generate_prbs` / `bits_to_symbols` / `symbols_to_bits` / `decide` / `add_awgn` /
  `ber_theory_qam` / `rc_filter` / `pulse_shape` / `laser_phase_noise` / `mimo_lms` / `mimo_lms_fse`
- `fiber.py` — `gaussian_pulse` / `sech_pulse` / `dispersion_step` / `nonlinear_step` /
  `propagate_ssfm`(Split-Step)/ `edfa` / `ase_psd`

`python _common/comm.py` を実行すると、共通ライブラリの自己テスト(PRBS 平衡性・QAM 往復・
BER 解析解一致)が走ります。

## 問題一覧

### 第1週: QAM 送受信の基礎

| No. | 問 | 問題名 | フォルダ |
| --- | --- | ------ | -------- |
| 01 | 1-1 | PRBS の生成と強度スペクトル | [01_prbs15](problems/01_prbs15/) |
| 02 | 1-2 | PRBS → QAM マッピング(グレイ符号・規格化) | [02_qam_mapping](problems/02_qam_mapping/) |
| 03 | 1-3 | QAM への AWGN 付加(SNR=20 dB) | [03_qam_awgn](problems/03_qam_awgn/) |
| 04 | 1-4 | 受信シンボルの判定(decision) | [04_qam_decision](problems/04_qam_decision/) |
| 05 | 1-5 | QAM デマッピング(シンボル→ビット) | [05_qam_demapping](problems/05_qam_demapping/) |
| 06 | 1-6 | BER の測定と処理時間計測 | [06_ber](problems/06_ber/) |

### 第2週: BER 曲線とパルス整形

| No. | 問 | 問題名 | フォルダ |
| --- | --- | ------ | -------- |
| 07 | 2-1 | BER の SNR 依存性(解析解と比較) | [07_qam_ber_curve](problems/07_qam_ber_curve/) |
| 08 | 2-2 | レイズドコサイン スペクトル整形 | [08_raised_cosine](problems/08_raised_cosine/) |
| 09 | 2-3 | 整形信号への複素 AWGN 付加 | [09_rc_awgn](problems/09_rc_awgn/) |
| 10 | 2-4 | ダウンサンプリング・復調・BER | [10_rc_downsample_ber](problems/10_rc_downsample_ber/) |
| 11 | 2-5 | 整形信号の BER-SNR 依存性 | [11_rc_ber_curve](problems/11_rc_ber_curve/) |
| 12 | 2-6 | 整形信号の時間遅延(端数遅延) | [12_delay](problems/12_delay/) |

### 第3週: 位相雑音・偏波・MIMO

| No. | 問 | 問題名 | フォルダ |
| --- | --- | ------ | -------- |
| 13 | 3-1 | レーザ位相雑音の生成と線幅(分散の証明付き) | [13_phase_noise](problems/13_phase_noise/) |
| 14 | 3-2 | 位相雑音付き QAM コンステレーション | [14_phase_noise_qam](problems/14_phase_noise_qam/) |
| 15 | 3-3 | 単一タップ MIMO + LMS 適応等化 | [15_single_tap_lms](problems/15_single_tap_lms/) |
| 16 | 3-4 | 両偏波 QAM シンボル列の生成 | [16_dual_pol](problems/16_dual_pol/) |
| 17 | 3-5 | 偏波回転(SU(2) 行列) | [17_pol_rotation](problems/17_pol_rotation/) |
| 18 | 3-6 | 2×2 MIMO 等化と BER(1 sps) | [18_mimo_ber](problems/18_mimo_ber/) |
| 19 | 3-7 | 2×2 MIMO 等化(2 sps、タップ数比較) | [19_mimo_2sps](problems/19_mimo_2sps/) |

### 第4週: 光ファイバ伝搬

| No. | 問 | 問題名 | フォルダ |
| --- | --- | ------ | -------- |
| 20 | 4-1 | 波長分散によるパルス広がり(解析解導出) | [20_dispersion](problems/20_dispersion/) |
| 21 | 4-2 | 非線形効果(SPM)による位相変化(解析解導出) | [21_nonlinear](problems/21_nonlinear/) |
| 22 | 4-3 | 基本ソリトン(N=1)の伝搬 | [22_soliton](problems/22_soliton/) |
| 23 | 4-4 | EDFA 前置増幅構成の BER vs Pin | [23_edfa_preamp](problems/23_edfa_preamp/) |
| 24 | 4-5 | 損失+EDFA 多中継伝送の BER vs Pin | [24_multispan_loss](problems/24_multispan_loss/) |
| 25 | 4-6 | 損失+分散+分散補償の BER vs Pin | [25_multispan_cd](problems/25_multispan_cd/) |
| 26 | 4-7 | 損失+分散+非線形(Split-Step)の BER vs Pin | [26_multispan_nl](problems/26_multispan_nl/) |

## 使い方

各問題フォルダで解答を実行できます(図は同フォルダに PNG で保存されます)。

```bash
python problems/NN_問題名/solution.py
```

> Windows のコンソールで日本語出力が文字化けする場合は、`PYTHONIOENCODING=utf-8` を付けて実行してください。
> 図中のラベルは、日本語フォント非依存にするため英語で記述しています(解説本文は日本語)。
