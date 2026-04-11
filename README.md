# MAX-CUT on G22 — CIM, CAC, and SA implementations

Python/Numba implementations and a fair comparison of three Ising-inspired
MAX-CUT solvers on the **G22** instance (N=2000, K=19990) from the
[Stanford G-Set benchmark](https://web.stanford.edu/~yyye/yyye/Gset/).

This repository reproduces two physics-inspired solvers (**CIM** and **CAC**)
from recent papers and compares them against a classical baseline
(**Simulated Annealing**).

## Methods

| Method | Source | Idea |
|---|---|---|
| **CIM** | Inoue & Yoshida, *Opt. Commun.* **522**, 128642 (2022) | Traveling-wave model of a coherent Ising machine based on a fiber loop with a pulse-pumped phase-sensitive amplifier. |
| **CAC** | Leleu et al., *Comm. Phys.* **4**, 266 (2021) | Chaotic Amplitude Control: per-pulse error variables, dynamic target amplitude, and a time-varying coupling ramp escape local minima. |
| **SA**  | Kirkpatrick et al., *Science* **220**, 671 (1983) | 1-flip exponential-cooling simulated annealing (baseline). |

All solvers maximize the number of cut edges on the **G22** graph, whose
known-best solution is **13359**.

## Results (100 trials each on G22)

| Method | Mean | Best | Worst | Std | Wall-clock |
|---|---|---|---|---|---|
| CIM | 13275.3 | 13326 | 13220 | 20.5 | **3.3 s** |
| CAC | 13284.8 | **13358** | 13214 | 25.8 | 206.5 s |
| SA  | 13224.8 | 13314 | 13048 | 50.1 | 200.3 s |

CAC reaches within **1 edge of the known optimum** (13358 vs 13359) and has
the best mean. CIM is the fastest by a wide margin while still reaching 99.75%
of optimum. SA is the weakest here — it plateaus around 13300 and has much
higher variance.

Generated figures:

- `results/compare_histogram.png` — distribution of cut values per method
- `results/compare_running_best.png` — running best cut vs trial count
- `results/compare_bar.png` — mean and best cut bar chart

### Reproducing the paper baselines

- **CIM** reproduction: 100 trials achieve mean **13275.3**, matching the
  paper's mean of 13275 (Inoue & Yoshida 2022, Fig. 8). Best achieved
  **13326**, slightly above the paper's best of 13321.
- **CAC** reproduction: The paper reports best 13359 with
  p₀ = 0.11 single-run success rate **on FPGA**. On CPU with ~200 s per 100
  trials, we reach best **13358** (1 cut short of optimum). Reaching the
  paper's p₀ would require 1–2 orders of magnitude more outer steps.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install Python dependencies
uv sync
```

Dependencies: `numpy`, `scipy`, `numba`, `wandb`, `matplotlib`, `pymupdf`.
Python 3.13+.

## Usage

### Single-run CIM (with Weights & Biases per-round logging)

```bash
uv run python CIM.py
```

### 100-trial CIM (parallel via Numba prange)

```bash
uv run python CMI_multi_run.py
```

### 100-trial CAC

```bash
uv run python CAC.py
```

### Baseline SA

```bash
uv run python SA.py
```

### Fair comparison (generates figures in `results/`)

```bash
uv run python compare.py
```

## Implementation notes

- **Numba JIT + `prange`**: the `_simulate_cim_batch` and `_simulate_cac_batch`
  functions are ahead-of-time compiled with Numba's parallel mode, so 100
  trials are distributed across CPU cores and finish in seconds.
- **Sparse coupling matrix**: G22 has only ~1% density so the coupling matrix
  is stored as scipy CSR. The matvec `J @ c` is hand-coded inside the JIT
  function for minimum overhead.
- **Verification**: `scripts/verify.py` independently cross-checks the cut
  count from the edge list vs the adjacency list to catch any bugs in the
  solver's internal cut tracking.

## Repository layout

```
.
├── CIM.py              # Inoue & Yoshida 2022 CIM simulator + single-trial entry point
├── CAC.py              # Leleu et al. 2021 CAC simulator + multi-trial entry point
├── SA.py               # Simple 1-flip simulated annealing baseline
├── CMI_multi_run.py    # CIM 100-trial parallel runner + wandb logging
├── compare.py          # Fair comparison of all three methods (produces figures)
├── scripts/
│   └── verify.py       # Independent cut-count verification
├── input/
│   └── G22.txt         # G22 instance from G-Set benchmark
└── results/            # Output figures from compare.py
```

## Unit scaling note for CIM

The noise variance formula in Inoue & Yoshida 2022 Eq. (6) is written as
`σ² = (2-η) G / 4 · BW` in "one-photon energy units" while the saturation
coefficient `γ = 42.09 W⁻¹` in Eq. (14) uses physical Watts. The unit
bridge between these two conventions requires an additional factor of ℏω.
`CIM.py` multiplies `σ²` by `photon_energy_J = 1.28e-19` (ℏω at 1550 nm) to
reconcile the two conventions. Without this correction, the noise
overwhelms the signal and the simulator gets stuck in a pathological state.

## License

MIT — see `LICENSE`.

## Citations

If you use this repository, please cite the original papers.

```bibtex
@article{Inoue2022,
  author  = {Kyo Inoue and Kazuhiro Yoshida},
  title   = {Traveling-wave model of coherent Ising machine based on fiber loop with pulse-pumped phase-sensitive amplifier},
  journal = {Optics Communications},
  volume  = {522},
  pages   = {128642},
  year    = {2022},
  doi     = {10.1016/j.optcom.2022.128642}
}

@article{Leleu2021,
  author  = {Timothée Leleu and Farad Khoyratee and Timothée Levi and Ryan Hamerly and Takashi Kohno and Kazuyuki Aihara},
  title   = {Scaling advantage of chaotic amplitude control for high-performance combinatorial optimization},
  journal = {Communications Physics},
  volume  = {4},
  pages   = {266},
  year    = {2021},
  doi     = {10.1038/s42005-021-00768-0}
}
```
