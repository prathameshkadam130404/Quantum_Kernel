# Bell-Decomposition Quantum Kernels for Land Cover Classification

Source code, experiment scripts, result metadata, and submission package for:

> **"Bell-Decomposition Quantum Kernels: A Multi-Scale Benchmark on
> Earth Observation and Beyond"** — Prathamesh Balasaheb Kadam (lead;
> primary corresponding), Prajwal S. Gaikwad, Shreyas Subhash Raut,
> Tejas Uttam Shitole; all four listed as corresponding authors.
> AISSMS Institute of Information Technology, Pune.
> Submitted to *Quantum Machine Intelligence* (Springer).

Paper: [`submission/sn-article.pdf`](submission/sn-article.pdf)
(LaTeX source: [`submission/sn-article.tex`](submission/sn-article.tex)).

---

## Headline result

We introduce the **Bell-decomposition** family of two-qubit quantum-kernel
feature maps, unifying the singlet-only **SRQFM** and the four-Bell-sector
**BSCM** under an adjustable prior.  Within the canonical two-body
Pauli-rotation feature-map framework of
[Suzuki et al. 2020](https://doi.org/10.1007/s42484-020-00020-y) (QMI),
the construction is a non-negative-prior weighted sum
$H_{ij}=\sum_k p_k\,w_k\,2\,|B_k\rangle\!\langle B_k|$ over the four
Bell-state projectors.  This admits two closed-form structural
identities specific to the Bell-projector subspace:

- A tight prior-independent bound $|\alpha|\le 1/2$ on every per-pair
  Pauli coefficient.
- The identity $\alpha_{ZZ}\equiv 0$ for the uniform prior, making
  BSCM-uniform strictly Pauli-orthogonal to any SRQFM-style $ZZ$
  generator.

A single hyperparameter $\tau$, locked once from a 200-sample holdout
at $n=8$, transfers operationally to $n=14$ and $n=16$ without
re-tuning.

### Four empirical findings

| Claim | Headline numbers |
|---|---|
| **(i) Architectural advantage** at $n=16$, $N=10{,}000$ | Bell-decomposition projected kernels add $+0.167$–$+0.170$ macro-F1 over commuting $ZZ$-PQK on So2Sat LCZ42 (17 classes) and $+0.097$–$+0.108$ on EuroSAT (10 classes); Holm-corrected Wilcoxon $p=0.0078$ on both.  Exact McNemar on the test-fold predictions: pooled $p \ll 10^{-100}$. |
| **(ii) Domain-agnostic** | Positive gap on every split across four UCI tabular datasets (Digits, Breast Cancer, Letter Recognition, Wine); largest gap $+0.646$ accuracy on the 26-class Letter task, whose 16 features match $n=16$ exactly.  Same sign on a transverse-field Ising-model phase-classification benchmark. |
| **(iii) Noise-robust** | Under density-matrix simulation at $n=8$ with depolarising noise, gap stays at $+0.133$–$+0.136$ macro-F1 across $p\in\{0, 0.001, 0.005, 0.01\}$ (Wilcoxon $p=6.1\times 10^{-5}$). |
| **(iv) Concentration on real data** | A 14-qubit BSCM build at $N=800$ and a 16-qubit depth scan over $L\in\{2,\dots,10\}$ at $N=2{,}000$ reproduce the [Thanasilp et al. 2024](https://doi.org/10.1038/s41467-024-49287-w) concentration signature on real multi-class remote-sensing data. |

### Honest negatives

**We make no claim of quantum advantage.**  Tuned classical RBF-SVM ties
or leads on every dataset and scale we test, consistent with recent
bandwidth-tuning convergence theory (Canatar et al. 2023, TMLR;
[Flórez-Ablan, Roth and Schnabel 2025](https://doi.org/10.1088/2058-9565/ade7ad),
Quantum Sci. Tech.) and with the empirical findings of
[Bowles et al. 2024](https://arxiv.org/abs/2403.07059) and
[Schnabel & Roth 2025](https://doi.org/10.1007/s42484-025-00273-5).  The
within-quantum architectural gap (BSCM family vs commuting $ZZ$) is
reported as the principal contribution.

To our knowledge, the $n=16$, $N=10{,}000$, $K\in\{10, 17\}$-class
Earth-observation benchmark in this paper is the largest fixed-protocol
multi-class quantum-kernel benchmark on remote-sensing data published
to date.

---

## Repository layout

```
src/                              kernel implementations
  bscm_kernel.py                    BSCM family (uniform, phi_only, psi_only)
  srqfm_kernel.py                   SRQFM (singlet-only ZZ) — PQK variant
  srqfm_fidelity_kernel.py          SRQFM — fidelity variant
  attention_kernel.py               Fisher feature selection used by E26/E32
  feature_extraction.py             IncrementalPCA + physics-feature pipeline
  kernel_target_alignment.py        KTA + variants
  classical_kernels.py              RBF / RF / poly baselines
  classifiers.py                    SVM / GridSearchCV wrappers
  bandwidth.py                      PQK bandwidth selection
  expressibility.py                 circuit expressibility diagnostic
  dequantization.py                 RFF dequantisation (E6)
  kernel_concentration.py           off-diag / within-σ / effective rank

experiments/                      paper-cited experiment scripts
  exp_e26_fair_srqfm.py               E26: 8-qubit C-tuned benchmark (Tab. 3, §5.2.1)
  exp_e32_bscm_fidelity_so2sat.py     E32: BSCM family at n=8 (Tab. 4, §5.2.2)
  exp_e37_pqk_scaling.py              E37: 16q depth sweep (Tab. 10, Fig. 7, §5.4.3)
  exp_e38_max_data_baselines.py       E38: 16q classical baselines (Tab. 11, §5.5)
  exp_e38_max_data_pqk.py             E38: 16q BSCM-PQK at N=10,000 (Tab. 11, Fig. 8, §5.5)
  exp_e38_max_data_standard_pqk.py    E38: 16q vanilla ZZ-PQK (Tab. 11, §5.5)
  exp_e42_ae16_pqk.py                 E42: AE-16 deep features (Tab. 12, §5.6.1)
  exp_e43_generalization_pqk.py       E43: 4-UCI domain generalisation (Tab. 6, Fig. 5, §5.3)
  exp_e44_quantum_advantage.py        E44: synthetic FQK advantage testbed (§5.6.2)
  exp_e46_spin_chain_advantage.py     E46: negative-control sanity check (§5.6.4)
  exp_e47_xgboost_baseline.py         E47: XGBoost row of Tab. 11 (§5.5)
  exp_e48_noise_density_matrix.py     E48: density-matrix noise sweep (Tab. 13, §5.6.3)
  exp_e49_tfim_pqk.py                 E49: TFIM phase classification (Tab. 7, §5.3.1)

  audit_e38_cvkta_bandwidth.py        CV-KTA bandwidth audit (§3.3)
  audit_e38_significance.py           Holm-Wilcoxon p-values for Tab. 11 (§3.5)
  audit_e38_perclass_mcnemar.py       Per-class CIs + McNemar for Tab. 11 (§3.5, §5.5)
  audit_pool_variance_rbf.py          Cross-pool RBF variability (§3.1)
  build_e38_summary.py                Consolidate E38 cells (Tab. 11) into one JSON

scripts/                          utility / data-pipeline scripts
  bscm_phase2_lockdown.py             τ-lock at n=8 (yields τ=0.25)
  bscm_phase2_lockdown_extended.py    extended τ-grid boundary check (§3.5)
  bscm_14q_test.py                    B2: 14-qubit BSCM regression
  extract_so2sat_10k.py               Build So2Sat physics-Fisher-16 pool
  extract_so2sat_ae16.py              AE-16 bottleneck extractor (E42)
  generate_tfim_dataset.py            TFIM 16-site ground-state features (E49)
  eurosat_data.py                     EuroSAT 13-band loader
  extract_physics_features.py         Physics-8 / 16 extractor

tests/                            machine-verified appendix claims
  test_proofs.py                      Theorem 1, Lemma 1, Propositions 1–3
  test_bscm_unitary.py                Unitary structure assertions
  test_pipeline.py                    End-to-end pipeline sanity

submission/                       paper package
  sn-article.tex                      LaTeX source (Springer sn-jnl class)
  sn-article.pdf                      Compiled manuscript (39 pages)
  sn-article.bbl                      BibTeX output
  refs.bib                            Bibliography database
  sn-jnl.cls                          Springer Nature journal class
  sn-mathphys.bst                     Numbered math/physics reference style
  cover_letter.{tex,pdf}              Cover letter for QMI submission
  README.md                           Submission-package contents
  figures/fig_*.pdf                   8 cited figures, vector PDF only

results/                          paper-essential result metadata
  bscm/locked_tau{,_extended}.json    τ-lockdown results
  e38_max_data/                       E38 caches + summary JSONs + audit outputs
    cvkta_spotcheck.json                CV-KTA bandwidth audit
    e38_combined_summary.json           Consolidated Tab. 11 cells
    e38_significance.json               Holm-Wilcoxon p-values
    e38_perclass_mcnemar.json           Per-class CIs + McNemar
    max_summary_bscm_sectors.json       BSCM family means
    standard_pqk_summary.json           Standard-ZZ-PQK means
    baselines_summary.json              Classical baselines
  e42_ae16_pqk/summary.json           E42 deep-feature numbers
  e43_generalization/                 E43 UCI domain-generalisation JSON
  e44_quantum_advantage/              E44 testbed result
  e46_unbiased_audit/                 E46 negative control
  e47_xgboost/xgb_baseline.json       E47 XGBoost row
  e48_noise_dm/                       E48 density-matrix noise sweep
  e49_tfim/tfim_phase_results.json    E49 TFIM cells
  pool_variance_rbf/                  Cross-pool RBF variance audit

config.py, requirements.txt, setup_data.py
```

Large kernel matrices (`cache_*.npz`) are not included; they are
regenerated by re-running the corresponding experiment script.

---

## Reproducing the paper

1. Install dependencies (PennyLane + scikit-learn + scipy + xgboost + h5py):
   ```bash
   pip install -r requirements.txt
   ```

2. Download the raw datasets (one-time):
   - So2Sat LCZ42: <https://mediatum.ub.tum.de/1483140>
   - EuroSAT (all-bands): <https://github.com/phelber/EuroSAT>
   - Run `python setup_data.py` to extract the feature caches.

3. Reproduce the headline E38 table (Tab. 11 of the paper):
   ```bash
   # One-time Bloch-vector extraction at n=16, N=10,000 (slow):
   python experiments/exp_e38_max_data_pqk.py
   python experiments/exp_e38_max_data_standard_pqk.py
   python experiments/exp_e38_max_data_baselines.py
   python experiments/exp_e47_xgboost_baseline.py

   # Consolidate Tab. 11 cells:
   python experiments/build_e38_summary.py

   # Audits cited in the paper (read cached Gram matrices, fast):
   python experiments/audit_e38_significance.py
   python experiments/audit_e38_perclass_mcnemar.py
   python experiments/audit_e38_cvkta_bandwidth.py
   ```

4. Reproduce the domain-generalisation table (Tab. 6, E43):
   ```bash
   python experiments/exp_e43_generalization_pqk.py
   ```

5. Reproduce the noise sweep (Tab. 13, E48):
   ```bash
   python experiments/exp_e48_noise_density_matrix.py
   ```

6. Reproduce the TFIM phase-classification table (Tab. 7, E49):
   ```bash
   python scripts/generate_tfim_dataset.py
   python experiments/exp_e49_tfim_pqk.py
   ```

7. Verify every Theorem / Lemma / Proposition claim in the appendix:
   ```bash
   python -m pytest tests/test_proofs.py -v
   ```

Approximate single-CPU wall-clock (no GPU) on a modern desktop with
PennyLane `lightning.qubit`:

| Experiment | Time |
|---|---|
| E26 ($n=8$, $N=2{,}000$, 5 splits) | ~4 h |
| E32 ($n=8$, $N=1{,}800$, 3 BSCM priors) | ~8 h |
| E37 ($n=16$, depth sweep, 5 splits) | ~18 h |
| E38 ($n=16$, $N=10{,}000$, 10 splits, all 3 BSCM priors) | ~36 h Bloch + minutes/kernel SVM |
| E42 (AE-16, $n=16$) | ~3 h |
| E43 (4 UCI datasets, $n=16$) | ~1.5 h |
| E48 (density-matrix noise, $n=8$, 4 noise levels × 3 seeds) | ~1.5 h |
| E49 (TFIM phase classification) | ~30 min |
| B2 (14-qubit BSCM build) | ~12 h |
| All E38 audits | ~15–30 min each (no quantum sim) |

Cached Gram matrices live in `results/<exp>/cache_*.npz`; experiment
scripts skip the quantum step when the cache exists.  Delete the cache
to force a re-run.

---

## Citation

```bibtex
@article{kadam2026bell,
  title    = {Bell-Decomposition Quantum Kernels: A Multi-Scale
              Benchmark on Earth Observation and Beyond},
  author   = {Kadam, Prathamesh Balasaheb and Gaikwad, Prajwal S. and
              Raut, Shreyas Subhash and Shitole, Tejas Uttam},
  journal  = {Quantum Machine Intelligence},
  year     = {2026},
  note     = {Under review}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Datasets used

- **So2Sat LCZ42** — Zhu et al., *IEEE Geosci. Remote Sens. Magazine* 2020,
  <https://doi.org/10.1109/MGRS.2020.2964708>
- **EuroSAT** — Helber et al., *IEEE J-STARS* 2019,
  <https://doi.org/10.1109/JSTARS.2019.2918242>

## Contact

All four authors are corresponding authors.  Primary corresponding
author for submission-related correspondence:

- Prathamesh Balasaheb Kadam — `prathamesh.kadam@aissmsioit.org`

Additional corresponding authors:

- Prajwal S. Gaikwad — `prajwal.gaikwad@aissmsioit.org`
- Shreyas Subhash Raut — `shreyas.raut@aissmsioit.org`
- Tejas Uttam Shitole — `tejas.shitole@aissmsioit.org`
