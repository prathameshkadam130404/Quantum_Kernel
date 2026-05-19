# Quantum Machine Intelligence — submission procedure

Step-by-step guide for submitting *"Bell-Decomposition Quantum Kernels: A Multi-Scale Benchmark on Earth Observation and Beyond"* to the Springer Nature journal **Quantum Machine Intelligence** (QMI).

Last verified against the official QMI submission guidelines on 2026-05-19.

---

## 1. Pre-flight checklist

Before opening the submission portal, confirm the items below — every one of them is a hard requirement of the QMI guidelines and the Springer Nature editorial policy:

- [x] Abstract is 229 words (QMI window: **150–250 words**).
- [x] Keywords: 6 entries (QMI window: **4–6**).
- [x] Heading literally reads `Statements and Declarations`.
- [x] All nine declarations are present:
      Funding · Competing interests · Ethics approval ·
      Consent to participate · Consent for publication ·
      Data availability · Materials availability ·
      Code availability · Author contributions.
- [x] AI/LLM disclosure is in §3.2 of the Methods (Springer Nature requires generative-AI use to be disclosed in the Methods section).
- [x] ORCID iDs are visible on the title page for all four authors.
- [x] The corresponding author's email is on the title page.
- [x] Bibliography is compiled with BibTeX (`sn-mathphys.bst`) — produces author-year citations in the text per QMI rules.
- [x] All eight cited figures are present as vector PDF inside `figures/`.
- [x] Cover letter compiled (`cover_letter.pdf`).
- [x] `sn-article.pdf` re-compiles end-to-end (`pdflatex → bibtex → pdflatex → pdflatex`) with no LaTeX errors and no undefined references.

If any of the above is `[ ]`, fix it before continuing.

---

## 2. The portal

QMI uses the **Springer Nature New Submission System** (NSS), not the older Editorial Manager interface:

> **Portal URL:** https://submission.nature.com/new-submission/42484/3
> *(`42484` is QMI's journal ID. `/3` is the manuscript-type slot — verify it still routes to "Original research article" before submitting; QMI may rotate slot numbers across manuscript types.)*

You can also reach the same portal indirectly from the journal homepage
(https://link.springer.com/journal/42484) → *Submit your manuscript* → it will redirect to the NSS URL above.

### Account requirements
- Each author should have an ORCID iD linked to a Springer Nature account.
  For our paper this is already done (the four iDs are in the title block).
- The submitting author (corresponding author: **Prathamesh Balasaheb Kadam**) signs in with their personal Springer Nature account.

---

## 3. Files to upload, in order

The portal lets you upload multiple files and assigns each a *role*. Upload these files in this order and tag each with the role indicated:

| Order | Filename                | Role to select in the portal              |
|-------|-------------------------|--------------------------------------------|
| 1     | `cover_letter.pdf`      | *Cover letter*                             |
| 2     | `sn-article.pdf`        | *Manuscript* (the version sent to reviewers) |
| 3     | `sn-article.tex`        | *Manuscript source* (LaTeX source)         |
| 4     | `refs.bib`              | *Supplementary file* (Bibliography source) |
| 5     | `sn-article.bbl`        | *Supplementary file* (BibTeX output, backup) |
| 6     | `sn-jnl.cls`            | *Supplementary file* (LaTeX class)         |
| 7     | `sn-mathphys.bst`       | *Supplementary file* (BibTeX style)        |
| 8–15  | `figures/fig_*.pdf` (×8) | *Figure* — one per upload, in the order they are cited in the text |

QMI accepts LaTeX source bundles — you do **not** have to convert to .docx.
Figures must each be uploaded **as a separate file**, not embedded only as PDF page-references; the portal will re-link them against the `\includegraphics{...}` calls in the .tex source.

> **Do not** include the local `D:/...` paths anywhere; the portal expects bare filenames matching `\includegraphics{figures/fig_*.pdf}`.

---

## 4. What you'll fill in inside the portal

| Field                       | What to enter                                                                                                                |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Article type                | *Original research article*                                                                                                  |
| Section                     | (None required)                                                                                                              |
| Title                       | Copy verbatim from the title page                                                                                            |
| Short title (running head)  | "Bell-Decomposition Quantum Kernels"                                                                                          |
| Abstract                    | Paste the abstract from §0 — single paragraph, 229 words                                                                     |
| Keywords                    | Paste 6 keywords exactly as in the manuscript                                                                                |
| Authors                     | Add all four with **affiliations + ORCID iDs**; the portal will request each author's email so co-authors should be invited and asked to add theirs |
| Corresponding author        | Prathamesh Balasaheb Kadam (the portal marks this with an asterisk)                                                          |
| Funding                     | "No external funding was received for this study."                                                                            |
| Competing interests         | "The authors declare no competing financial or non-financial interests."                                                     |
| Data availability statement | Copy from §Statements-and-Declarations of the manuscript                                                                     |
| Code availability statement | Copy from §Statements-and-Declarations                                                                                       |
| Author contributions        | Copy the four-paragraph block from §Statements-and-Declarations                                                              |
| Suggested reviewers (opt.)  | You may suggest up to 4. Avoid co-workers and recent collaborators. Reasonable suggestions for this paper: an author on Schnabel & Roth 2025 (QMI), Bowles et al. 2024, Incudini et al. 2025 (QMI), Thanasilp et al. 2024 |
| Non-preferred reviewers (opt.) | Authors of any recent competing manuscript only (none for us)                                                              |
| AI/LLM use                  | Tick the checkbox; the portal asks for the disclosure text — copy the *Use of generative-AI tools* paragraph from §3.2 of the manuscript |

> Some fields are duplicated between the portal metadata and the `Statements and Declarations` block in the manuscript — keep both consistent.

---

## 5. After you click "Submit"

1. The portal returns a manuscript number (typically of the form `QMIN-D-26-NNNNN`).
   **Record it** — every subsequent email from QMI will reference it.
2. The submitting author receives a confirmation email within an hour.
3. Each co-author receives a separate email asking them to confirm authorship
   and (if not yet on ORCID) to link their iD. **Forward this to the co-authors
   and ask them to act on it within a few days** — submission stays in a
   "Pending Author Confirmation" state until they do.
4. The handling editor screens the paper (~1–3 days). One of three things
   happens:
   - **Sent for review** → status changes to *Under Review* and reviewers
     are invited.
   - **Desk rejection** → notification within ~1–2 weeks.
   - **Request for minor format fixes** → revise and resubmit through the
     portal (status: *Reviewer Reports Required, action: author*).
5. First-round reviews typically arrive **8–12 weeks** after the desk
   screening (QMI median per *researcher.life* metrics).
6. Decisions: Accept / Minor revision / Major revision / Reject. For QMI the
   modal first-round outcome on technical papers is *Major revision*.

---

## 6. Common pitfalls — read this list once

- **ORCID confirmation lag.** Co-authors need to actively confirm their iD inside the portal. The manuscript will not move to review until all four have done so.
- **Figure naming.** Do not rename figures during upload; the portal will fail to bind them to `\includegraphics{figures/fig_*.pdf}` in the .tex.
- **Bibliography style.** The portal will re-compile the LaTeX source server-side; if `sn-mathphys.bst` is missing from the upload, the back-end produces "ref [?]" markers and the editor will bounce the paper. **Always upload `sn-mathphys.bst` together with `refs.bib`.**
- **`.aux`, `.log`, `.toc` files.** Do not upload LaTeX build artefacts; only the source files listed in §3.
- **Anonymous-vs-open review.** QMI is **single-blind** (reviewers see the authors, the authors do not see the reviewers). The manuscript should *not* be anonymised — keep the title-page author block intact.
- **Preprints.** Posting to arXiv before submission is allowed by QMI and Springer Nature's open-access policy. If you plan to do this, do it *before* submission and add the arXiv ID in the cover letter under *Originality and exclusivity*.
- **AI/LLM disclosure consistency.** The portal asks for an AI disclosure independently of the manuscript. Use the *same* text in both places to avoid an editorial-board flag for inconsistency.

---

## 7. After acceptance

After the paper is accepted you will receive two automated emails from
Springer Nature:

1. **Affiliation confirmation** — confirm institutional affiliation for the
   author-funded open-access discount, if your institution has one.
2. **Publishing model selection** — choose between (a) standard subscription
   publication or (b) Gold open access. As an Indian-affiliated paper at
   AISSMS IIT, OA fees may be partially or fully covered through Springer
   Nature's transformative-agreement programme; check before paying out of
   pocket.

A link to the typeset proofs will follow within 2–3 weeks; review them
carefully (especially numerical values in tables and the bibliography) and
return within the deadline given.

---

## 8. If you need to make a change before submission

Re-compile after any edit:

```bash
cd submission
pdflatex -interaction=nonstopmode sn-article.tex
bibtex sn-article
pdflatex -interaction=nonstopmode sn-article.tex
pdflatex -interaction=nonstopmode sn-article.tex
```

Confirm in `sn-article.log` that there are no `! LaTeX Error` lines and no
`LaTeX Warning: Reference ... undefined` warnings before re-bundling for
upload.
