# Submission Package
## Quantum Kernel Methods for Multi-Class Land Cover Classification on the So2Sat LCZ42 Dataset

**Target Journal:** Quantum Machine Intelligence (Springer)

---

## Contents

```
submission/
├── paper.tex              # Main manuscript (LaTeX source)
├── sn-jnl.cls             # Springer Nature journal class (YOU MUST ADD THIS)
├── figures/
│   ├── fig_learning_curves.pdf        # Fig 1: AGPQK vs RBF-SVM learning curves
│   ├── fig_concentration_scaling.pdf  # Fig 2: Off-diagonal variance vs qubit count
│   ├── fig_noise_degradation.pdf      # Fig 3: F1 vs depolarizing noise rate
│   ├── fig_srqfm_ablation.pdf         # Fig 4: SRQFM coupling variant ablation
│   ├── fig_topology_comparison.pdf    # Fig 5: Entanglement topology comparison
│   └── fig_ood_comparison.pdf         # Fig 6: OOD generalization bar chart
└── README.md              # This file
```

## Before Submission Checklist

- [ ] Place `sn-jnl.cls` in this directory (download from Overleaf or Springer)
- [ ] Compile: `pdflatex paper.tex && pdflatex paper.tex`
- [ ] Verify all 6 figures render correctly in the PDF
- [ ] Add GitHub/Zenodo repository URL in Data Availability section
- [ ] Final proofread for any remaining em-dash or encoding issues
- [ ] Remove `sn-template.zip` and `sn-template-tmp/` from parent directory

## Authors

1. Prathamesh Balasaheb Kadam (corresponding) — prathameshkadam130404@gmail.com
2. Shreyas Subhash Raut
3. Tejas Uttam Shitole
4. Prajwal S. Gaikwad

**Affiliation:** Department of Artificial Intelligence and Data Science,
AISSMS Institute of Information Technology, Pune, India
