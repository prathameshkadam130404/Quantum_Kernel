# Singlet-Gated Bell-Decomposition Quantum Kernels for Land Cover Classification

This repository contains the source code, experiment scripts, result
metadata, and manuscript for the paper:

> **"Singlet-Gated Bell-Decomposition Quantum Kernels for Land Cover
> Classification"** — Prathamesh Kadam (2026).
> Submitted to *Quantum Machine Intelligence* (Springer).

The paper is in [`submission/paper_revised.pdf`](submission/paper_revised.pdf)
(LaTeX source: [`submission/paper_revised.tex`](submission/paper_revised.tex)).

## Headline result

We evaluate two new families of quantum kernels — **SRQFM** (singlet-only
$ZZ$ generator) and **BSCM** (four Bell-weight family with adjustable
prior, $|\alpha|\le 1/2$, with $\alpha_{ZZ}\equiv 0$ for the uniform
preset) — plus a **Singlet-Gated BSCM (SG-BSCM)** correction, on
So2Sat LCZ42 (17-class) and EuroSAT (10-class) at 8, 14, and 16 qubits
with training pools up to $N=10{,}000$.

We make no claim of quantum advantage. Our quantum kernels match tuned
classical RBF-SVM at every scale we test but never decisively exceed
it — the expected behaviour of bandwidth-tuned PQKs under recent theory
([Heyraud et al. 2022](https://doi.org/10.1103/PhysRevA.106.052421);
[Slattery et al. 2025](https://arxiv.org/abs/2503.05602)).
The contributions are:

1. **SG-BSCM correction.** BSCM-uniform collapses to macro-F1
   $0.032\pm 0.008$ on EuroSAT physics-8 features. SG-BSCM, a one-line
   generator-level multiplier by the singlet weight $w_{\Psi^-}$,
   restores macro-F1 to $0.664\pm 0.027$ on par with SRQFM and tuned
   RBF-SVM. We give the mechanism a closed-form Gaussian bound
   (Proposition 4 in the paper).

2. **14-qubit concentration regression on real multi-class data.**
   Direct measurement of the
   [Thanasilp et al. 2024](https://doi.org/10.1038/s41467-024-49287-w)
   concentration signature on So2Sat physics-Fisher-14 at $N=800$,
   with off-diagonal mean $0.215\to 0.066$, KTA $0.59\to 0.42$, F1
   $0.483\to 0.418$.

3. **Largest fixed-protocol multi-class quantum-kernel benchmark on
   Earth-observation data we are aware of**: 16 qubits, $N=10{,}000$,
   5 stratified shuffle splits, $\tau=0.25$ locked from holdout.

## Repository layout

```
src/                           — kernel implementations
  bscm_kernel.py                  BSCM family (uniform, phi_only, psi_only) + SG-BSCM
  srqfm_kernel.py                 SRQFM (singlet-only ZZ) — PQK variant
  srqfm_fidelity_kernel.py        SRQFM — fidelity variant
  attention_kernel.py             AG-PQK (attention-gated PQK baseline)
  kernel_target_alignment.py      KTA + variants
  kernel_concentration.py         off-diag / within-σ / effective-rank diagnostics
  feature_extraction.py           So2Sat physics-8 / 16 feature extraction
  classical_kernels.py            RBF / RF / poly baselines
  classifiers.py                  SVM / GridSearchCV wrappers
  bandwidth.py                    PQK bandwidth selection
  expressibility.py               quantum circuit expressibility
  dequantization.py               RFF dequantisation experiment (E6)

experiments/                   — experiment scripts
  exp_e26_fair_srqfm.py           E26: fair C-tuned SRQFM benchmark
  exp_e32_bscm_fidelity_so2sat.py E32: BSCM family on So2Sat physics-8
  exp_e33_sg_bscm.py              E33: SG-BSCM on EuroSAT physics-8 + PCA-8
  exp_e33_sg_bscm_thresh0.py      E33-thresh0: gate-threshold ablation (pending)
  exp_e36_sg_bscm_pqk.py          E36: SG-BSCM-PQK (projected variant)
  exp_e37_pqk_scaling.py          E37: 16q depth scaling
  exp_e38_max_data_baselines.py   E38: 16q baselines at N=10,000
  exp_e38_max_data_pqk.py         E38: 16q BSCM-PQK at N=10,000
  exp_e38_max_data_standard_pqk.py E38: 16q vanilla ZZ-PQK at N=10,000
  exp_e39_noise_sgbscm.py         E39: noise robustness (pending)
  exp_e40_rho_sweep.py            E40: synthetic ρ-sweep verification of Prop 4 (pending)
  exp_e28_concentration_scaling.py E28: synthetic concentration scaling
  exp_e1_*, exp_e2_*, exp_e3_*, exp_e5_*, exp_e6_*, exp_e15_* — auxiliary

scripts/                       — utility scripts cited in paper
  comprehensive_comparison.py     master accuracy/F1 aggregator
  bscm_eurosat_fixed_pool.py      EuroSAT fixed-pool builder
  bscm_14q_test.py                14q BSCM kernel builder
  bscm_publication_analysis.py    publication statistics aggregator
  eurosat_data.py                 EuroSAT loader + physics indices
  extract_physics_features.py     So2Sat physics-8 / 16 extraction

tests/
  test_proofs.py                  20 machine-verified proof assertions

submission/                    — paper
  paper_revised.tex               main LaTeX source
  paper_revised.pdf               compiled paper
  figures/*.pdf, *.png            figures
  figures/*.py                    figure generation scripts
  sn-jnl.cls, sn-*.bst           Springer template

results/                       — result metadata (JSONs / CSVs / logs)
  Each subdirectory contains the summary.json / metrics.csv /
  log files that back specific tables in the paper.  Large kernel
  matrices (.npy / .npz) are NOT included; they are regenerated by
  re-running the corresponding experiment script.

config.py, requirements.txt, setup_data.py, run_all.py
```

## Reproducing the headline numbers

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the raw datasets (one-time):
   - So2Sat LCZ42 from <https://mediatum.ub.tum.de/1483140>
   - EuroSAT (all-bands) from <https://github.com/phelber/EuroSAT>
   - Run `python setup_data.py` to extract feature caches.

3. Reproduce a specific table from the paper:
   ```bash
   # Tab. 4 (E33 SG-BSCM rescue on EuroSAT physics-8):
   python -m experiments.exp_e33_sg_bscm

   # Tab. 6 (E37 depth scaling at 16 qubits):
   python -m experiments.exp_e37_pqk_scaling

   # Tab. 7 (E38 max-capacity benchmark, 16 qubits, N=10,000):
   python -m experiments.exp_e38_max_data_baselines
   python -m experiments.exp_e38_max_data_pqk
   python -m experiments.exp_e38_max_data_standard_pqk
   ```

4. Verify every mathematical claim in the appendices:
   ```bash
   python -m pytest tests/test_proofs.py -v
   ```
   Expected output: `20 passed`.

5. Aggregate cross-experiment accuracy / F1:
   ```bash
   python scripts/comprehensive_comparison.py
   ```

Wall-clock estimates: E33 ~10 h, E37 ~18 h, E38 (all three) ~30–36 h
on a single CPU using PennyLane's `lightning.qubit` simulator.

## Citation

If you use this code or framework, please cite:

```bibtex
@article{kadam2026sgbscm,
  title  = {Singlet-Gated Bell-Decomposition Quantum Kernels for Land Cover Classification},
  author = {Kadam, Prathamesh},
  journal = {Quantum Machine Intelligence},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT License (see [LICENSE](LICENSE)).

## Datasets used

- **So2Sat LCZ42** — Zhu et al., IEEE Geosci. Remote Sens. Magazine 2020,
  <https://doi.org/10.1109/MGRS.2020.2964708>.
- **EuroSAT** — Helber et al., IEEE J-STARS 2019,
  <https://doi.org/10.1109/JSTARS.2019.2918242>.

## Contact

Prathamesh Kadam — `prathameshkadam130404@gmail.com`
