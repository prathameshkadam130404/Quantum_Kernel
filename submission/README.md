# Submission package

**Manuscript title.**  Bell-Decomposition Quantum Kernels: A Multi-Scale Benchmark on Earth Observation and Beyond.

**Target journal.**  *Quantum Machine Intelligence* (Springer Nature), `sn-jnl` class with the `sn-mathphys` (numbered) reference style.

**Corresponding author.**  Prathamesh Balasaheb Kadam, AISSMS Institute of Information Technology, Pune, India.

---

## Contents

| File | Role |
|---|---|
| `sn-article.tex`        | Main manuscript source (the Springer Nature template default name). |
| `sn-article.pdf`        | Compiled manuscript (39 pages). |
| `sn-article.bbl`        | BibTeX-generated bibliography, uploaded as a backup. |
| `refs.bib`              | BibTeX database (22 entries). |
| `sn-jnl.cls`            | Springer Nature journal class (full template). |
| `sn-mathphys.bst`       | Bibliography style for math/physics journals. |
| `cover_letter.tex`      | Cover letter source. |
| `cover_letter.pdf`      | Cover letter (2 pages). |
| `figures/fig_*.pdf` (×8) | All figures cited in the manuscript, vector PDF. |

## Compile

From this directory (`submission/`):

```bash
pdflatex -interaction=nonstopmode sn-article.tex
bibtex sn-article
pdflatex -interaction=nonstopmode sn-article.tex
pdflatex -interaction=nonstopmode sn-article.tex
```

Expected output: `sn-article.pdf`, 39 pages, no `! LaTeX Error` and no undefined references.

## Cited figures (8, all PDF vector)

`fig_circuit.pdf`, `fig_concentration_hist.pdf`, `fig_depth_scaling.pdf`, `fig_domain_generalisation.pdf`, `fig_feature_correlation.pdf`, `fig_max_capacity.pdf`, `fig_pauli_heatmap.pdf`, `fig_topology.pdf`.

## Pre-upload checklist

- [x] Abstract = 229 words (QMI limit 150--250).
- [x] 6 keywords (QMI limit 4--6).
- [x] Heading `Statements and Declarations` present.
- [x] All required declarations included (Funding, Competing interests, Ethics, Consent to participate, Consent for publication, Data availability, Materials availability, Code availability, Author contributions).
- [x] Generative-AI tool use disclosed in Methods per Springer Nature policy.
- [x] Bibliography migrated to BibTeX (`refs.bib` + `sn-mathphys.bst`).
- [x] Full Springer `sn-jnl.cls` template installed.
- [x] 8 cited figures present as vector PDF.
- [x] Cover letter compiled.
- [x] ORCID iDs embedded in the title block for all four authors (rendered via the `orcidlink` package; clickable links to `https://orcid.org/<id>` are verified inside the PDF).
- [ ] **Co-author emails** to be added inside the QMI portal at submission time (the ORCID profiles have email set to private, so the addresses cannot be pulled programmatically). Only the corresponding author's email is on the title page; QMI's submission form will request the other three from the portal.

## Submission walkthrough

See `SUBMISSION_PROCEDURE.md` in this directory for the full QMI / Springer
Nature submission walk-through (portal URL, file order, metadata fields,
post-submission timeline, common pitfalls).
