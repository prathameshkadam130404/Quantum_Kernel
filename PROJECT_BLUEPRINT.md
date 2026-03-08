# Project Blueprint: Quantum-Sat Classification

## Section 1: Problem Statement
The project addresses the classification of satellite imagery into the Local Climate Zone (LCZ) scheme using the So2Sat LCZ42 dataset. This is a 17-class classification problem. The raw modalities available include Synthetic Aperture Radar (SAR) and multi-spectral Optical data.

**Task Hardness & Imbalance:**
The project identifies a steep 67:1 class imbalance within the dataset. To make quantum kernel simulation computationally tractable (usually bound by an $O(n^2)$ matrix evaluation cost), the dataset is heavily subsampled. The training subset uses 2,000 samples and the test subset uses 2,000 samples. Under this subsampling limit, the rarest class (Class 6: Lightweight Low-rise) only retains 22 samples in the training set. 

**Input Modalities:**
To fit within an 8-qubit quantum envelope, the raw images are feature-engineered using Principal Component Analysis (PCA) fitted on an independent 15,000-sample set. The pipeline limits the dimensionality to 8 continuous features (in $[0, \pi]$), split into 4 SAR features and 4 Optical features for the "fused" dataset. Classical SVMs with these PCA features can achieve 84% variance capture on Optical data alone, motivating the need for a rigorous quantum advantage protocol on the fused or SAR-only sets.

---

## Section 2: Theoretical Framework
This project's objective is to evaluate whether Quantum Machine Learning (QML) can provide a quantifiable advantage over classical approaches for remote sensing data, resting on several distinct theoretical pillars:

- **Geometric Difference ($g$):** Based on the framework by Huang et al. (2021). It quantifies the difference in the geometry of the data mapped by a quantum kernel versus a classical kernel. The project leverages $g(K_q, K_c) \gg 1$ as a necessary condition for quantum advantage.
- **Kernel Target Alignment (KTA):** Introduced by Cristianini et al. (2001), KTA measures how well a kernel correlates with the ideal target labels. High $g$ alongside a positive $\Delta\text{KTA} > 0$ indicates that the quantum geometric advantage is task-relevant, not just random expressivity noise.
- **Kernel Concentration:** The pipeline tests for exponential concentration (Thanasilp et al., 2024), where the off-diagonal entries of the quantum kernel matrix collapse to a constant. This limits generalizability. Projected Quantum Kernels (PQK) are intended to mitigate this by projecting global states to local Bloch vectors, reducing entanglement-induced feature dilution.
- **Covariant Architectures:** Drawing from group representation theory (e.g., Glick et al., 2024), the geometry of LCZ image patches is symmetric under 90-degree rotations and flips (the Dihedral group $D_4$). The pipeline models this via a $D_4$ Covariant feature map, evaluated as a standalone experiment. To guarantee group-averaging invariance, the implementation fits an orbit-augmented PCA subspace to the data, ensuring the encoded features $\theta(x)$ are invariant under $D_4$ transformations ($\theta(x) \approx \theta(gx)$) within a tolerance of $10^{-4}$.
- **Trained Fidelity Kernels (TFK):** The project trains the encoding data-scaling parameters ($\theta$) of the quantum circuit using the parameter-shift rule to maximize KTA. The implementation uses a "fixed anchor training subset" of 51 samples (3 per class) and evaluates all 1,275 intra-subset pairs continuously to suppress gradient variance, avoiding the stochastic divergence common in standard optimizers.

---

## Section 3: What the Code Implements
The `src/` directory contains the mathematical heart of the framework:

- **`config.py`**: The definitive record of hyperparameters. Defines 8 qubits, 2 ZZ feature map repetitions, fixed regularization parameters, TFK rates (`TFK_LR=0.02`), and specific fixed subsets for training blocks (e.g., 51 samples).
- **`quantum_kernels.py`**: Implements global Fidelity Quantum Kernels (FQK), Projected Quantum Kernels (PQK), Trained Fidelity Kernels (TFK), and Cross-Modal FQK (CM-FQK). The TFK optimizer maximized KTA over 75 epochs using the parameter-shift rule on a fixed anchor subset (51 samples, all 1,275 pairs).
- **`classical_kernels.py`**: Builds baseline classical kernels resulting in normalized comparisons: RBF, polynomial, Laplacian, linear, and notably a classical "tensor-product" baseline that processes SAR and Optical modalities entirely independently for multi-modal comparisons.
- **`kernel_target_alignment.py`**: Computes centered (HSIC-normalized) and class-weighted KTA (Rule I6). Class weighting adjusts the ideal target kernel to prevent dominant classes (e.g., water) from biasing the alignment.
- **`geometric_difference.py`**: Solves for $g$ using spectral norms over eigen-decompositions. The code deliberately flags the historical flaw of regularizing $K_c$ with an arbitrary precision $\lambda = 10^{-6}$ and recommends an adaptive $1/n$ regularization to prevent numerical explosions tied to classical rank deficiencies.
- **`kernel_concentration.py`**: Evaluates kernel performance loss by measuring the coefficient of variation (CV) and evaluating the off-diagonal means and effective rank trace quotients to detect if a chosen kernel provides flat structure logic.
- **`d4_covariant_kernel.py`, `d4_representation.py`, & `d4_augmented_pca.py`**: Calculates symmetric orbit permutations over the 8-component PCA space. To resolve invariance failures, `d4_augmented_pca.py` fits a PCA model on data augmented with all 8 $D_4$ transformations. This module is now part of a standalone experiment architecture.
- **`geometric_bound.py` & `expressibility.py`**: Calculates empirical mutual information between SAR/Opt PCA pairs, correlating the strength of empirical mutual information against the $g$ difference. Also tracks Haar-random expressibility matching bounds.

---

## Section 4: The Experiments Pipeline
The project is structured into a multi-phase computational pipeline running out of the `experiments/` directory:

- **Phase 2: `exp1_geometric_analysis.py`**  
  The central runner for computing $g$, KTA, $\Delta\text{KTA}$, and concentration metrics (CV, rank) across all modalities (SAR, Optical, Fused) for classical and quantum kernels (FQK, PQK, TFK). 
- **Phase 2.1: `exp_d4_standalone.py`**  
  Standalone evaluation of $D_4$ covariance on SAR data. Uses orbit-augmented PCA to guarantee symmetry preservation. Compares RBF, FQK, and D4-FQK within the same invariant feature space.
- **Phase 2.5: `exp10_controlled_mi.py`**  
  Attempts to establish a causal linkage between modality interplay and geometric power. Evaluates whether substituting Optical features with escalating mixing ratios of uniform random noise ($\alpha$) degrades $g$ and $\Delta\text{KTA}$ (using a SAR-corrupted null control).
- **Phase 2.75: `exp11_expressibility.py`**  
  Tests the mechanistic chain by measuring Haar-random expressibility (KL divergence) of the feature maps across modalities. Hypothesis: Higher MI $\to$ Higher expressibility $\to$ Higher $g$ $\dots$
- **Phase 3: `exp2_kernel_classification.py`**  
  Full SVM classification comparison across all 17 LCZ classes for quantum (FQK, PQK, TFK, CM-FQK) vs. classical kernels. Evaluates macro-F1 score as the primary metric.
- **Phase 4: `exp3_fewshot_curves.py`**  
  Measures model efficiency and scaling via Few-Shot learning curves ($N \in [50, 100, 200, 500, 1000, 2000]$). Calculates the Area Under the Learning Curve (AUC-LC) and exact sampling efficiencies compared to RBF models.
- **Phase 5: `exp4_topological_boost.py`**  
  Tests TDA synergy: Evaluating if persistent homology (Topological Data Analysis) complements internal quantum kernel encoding better than standard classical processing streams (using PCA-only, TDA-only, and PCA+TDA fusion vectors).
- **Phase 6: `exp5_equivariant_comparison.py`**  
  Trains explicit multi-layer states: Equivariant QNN, Non-equivariant QNN, Classical equivariant MLP, and Classical MLP over few-shot configurations, documenting parameter efficiency and validation accuracy.
- **Phase 7: `exp6_dequantization_defense.py`**  
  Subjecting the quantum models to mathematically rigorous dequantization sweeps via Random Fourier Features (RFF). Checks if the evaluated geometric advantages can be easily classically simulated.
- **Phase 8: `exp7_hardware_validation.py`**  
  Running subsets (10-sample pilot, 25-sample full) on real IBM Quantum hardware via Qiskit Runtime, capturing real-world device noise impacts on $g$ vs simulator expectations.
- **Phase 9: `exp8_multidataset.py`**  
  Cross-domain verification running the exact pipeline across the EuroSAT dataset to evaluate modality synergy requirements, predicting that $g$ should be lower when inter-modality complexity vanishes.
- **Phase 10: `exp9_full_ablation.py`**  
  A systematic 8-step subtraction of features (FQK $\to$ PQK $\to$ Class-weighting $\to$ TDA $\to$ Balanced $\to$ TFK $\to$ CM-FQK $\to$ RBF-baseline) to audit the explicit performance gain brought by each novel project addition.
- **Validation toolset (`scripts/`)**:  
  Ancillary scripts to verify TFK class balances (`verify_tfk_class_counts.py`), perform mid-flight kernel rank diagnostics (`diagnose_tfk.py`), and search for stable optimization gradients across learning rates (`tfk_lr_finder.py`).

---

## Section 5: Results So Far
Calculated evidence retrieved directly from the repository output configuration matrices and logs:

- **Baseline SAR KTA metrics:**
  - FQK Centered Weighted KTA: **0.1845**
  - PQK Centered Weighted KTA: **0.1467**
  - RBF (Classical) Centered Weighted KTA: **0.1308**
- **TFK Training Results (Final):**
  - Optimized over 75 epochs using the parameter-shift rule on 51 samples.
  - Initial (Epoch 1): KTA = 0.3216
  - Final (Epoch 75): KTA = 0.3871 (GO verdict from `validate_tfk_training.py`).
  - **Final Theta Statistics (8 parameters):**
    - Mean: 2.2877, Std: 0.1875
    - Min: 1.9192, Max: 2.4753
    - Distance penalty $||\theta - 1||$: 3.6805.
- **Geometric Difference values:**
  - Exact computed geometries exist in cached states dependent upon their static constraint parameter sweeps, notably varying substantially whether $10^{-6}$ or the proper adaptive $1/n$ regularization is employed.

*Note: Pending calculation of all remaining modalities' SVM boundaries and full-kernel scale testing for the `fused/tfk` matrices.*

---

## Section 6: Known Issues and Open Questions
Several vulnerabilities remain mathematically and structurally unproven across the repository boundaries:

- **The regularization constraint artifact:** The classical $g$ difference metric demonstrates severe numerical conditioning instability depending on the $\lambda$ application. Generating $g > 1000$ values when using a flat $10^{-6}$ coefficient appears to represent an empirical conditioning breakdown on near-degenerate RBF distributions rather than genuine quantum encoding supremacy. While the implementation favors $1/n$ regularization to stabilize this, verifying if this conforms strictly to theoretical bounds established by Huang et al. is open.
- **Fixed subset TFK generalization:** While replacing the diverging stochastic sampling of parameter shifts with an exact, 51-anchor evaluated gradient provides stable descent for Adam optimization (as proven by soaring KTA scores pushing 0.40 in earlier epochs), there is no guarantee that aligning to this isolated block structure universally generalizes out to the remaining $1,949$ dataset observations perfectly during the full validation evaluation. 
- **The $D_{4}$ invariance verification:** The $D_4$ group-averaging invariance failed (~0.045 error) when using standard PCA due to the non-orthogonal closure of the $M_g$ matrices in the PCA subspace. The fix (orbit-augmented PCA fitting) is implemented in `src/d4_augmented_pca.py`. Status: Fix implemented, invariance gate enforces max error < 1e-4.

---

## Section 7: What Is Still Pending
Resolutions required to move toward finalizing the pipeline outcomes structure:

1. **Conclusion of `exp1`:** The primary evaluation code has completed TFK generation on the fused database. Verification of final kernel values spanning 2,000 components and generalization away from the training subset is complete.
2. **D4 standalone experiment:** Evaluation of symmetry preservation and geometric amplification in the standalone D4 experiment (Phase 2.1). Compute estimate: ~50 hours.
3. **Execution of `exp10_controlled_mi.py`:** A causality proof utilizing noisy modulation injections to measure cross-modal geometry destruction correlations.
4. **Final matrix SVM comparisons:** Aggregation of the trained states back across `exp1` validation tests to see an $F1$ and Accuracy validation metric scoring summary.

---

## Section 8: Hardware and Compute Context
Quantum simulation algorithms for kernels inherently command intensive memory overhead scaling at $O(n^2)$ matrices. 

- **Environment:** Intel i5-13500HX, RTX 4050 GPU (6GB VRAM), running WSL2 with CUDA 12 support, bound to 16GB RAM constraint.
- **Simulation constraint:** Set to an 8-qubit local limit due to memory sizing.
- **Empirical Execution Timing:**
  - TFK Training computes true $1,275$-pair gradients concurrently. 
  - Logged evaluation rates in real-time display ~267.79 seconds per single epoch cycle step. 
  - Completing the 75-epoch routine requires just above 5.5 hours to formulate the encoded parameters without factoring in subsequent matrix resolution projections. 
  - Evaluating dense tensor generation over 2,000 arrays will consume considerable temporal overhead mirroring the evaluation epochs constraint.
