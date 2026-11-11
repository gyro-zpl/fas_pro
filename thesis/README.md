# Master's Thesis (UMAC FST)

**Title:** From Prompting to Structured Alignment: Individual, Population, and Temporal Boundaries of LLM Survey Simulation

**Author:** CHENYU YUAN
**Supervisor:** PENGYANG WANG
**Degree:** Master of Computer Science
**Department:** Computer Science, Faculty of Science and Technology, University of Macau
**Date:** April 2027

## Format

Follows the UMAC FST master's thesis template:

- A4 paper
- Times New Roman 12pt body, 1.5 line spacing
- Margins: left 1.5in (binding side), right 1in, top 1in, bottom 1in
- Chapter headings 14pt, captions 10pt
- Front matter: title page, abstract, TOC, list of figures, list of tables, list of abbreviations
- Page numbers: preliminary pages bottom center (TOC first page unnumbered); text pages top center; first page of each chapter/bibliography/appendix counted but unnumbered
- Table captions above tables; figure captions below figures
- Main body: Chapters 1--5, **10 pages**
- Back matter: bibliography, appendices A--C, vita (not listed in TOC)

## Build

```bash
cd thesis
tectonic main.tex
```

or with a standard TeX installation:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The compiled PDF is `main.pdf` (26 pages total, 10-page main body).

## Structure

```text
thesis/
├── main.tex              # preamble, title page, abstract, front matter
├── chapters/
│   ├── 01_intro.tex      # introduction + related work
│   ├── 02_method.tex     # data construction + method
│   ├── 03_results.tex    # individual + population results
│   ├── 04_temporal.tex   # temporal analysis + ablations
│   └── 05_conclusion.tex # limitations + conclusion
├── appendix/
│   ├── A_data.tex        # data audit, splits, sampling, features
│   ├── B_results.tex     # per-year table, ablations, cross-pressure
│   ├── C_fas.tex         # full FAS table, prompts, hyperparameters
│   └── D_second_family.tex # DeepSeek-R1-Distill-Llama-8B robustness check
├── tables/
│   ├── table2_pas.tex
│   └── table3_fas.tex
├── figures/
│   └── fig2_temporal_decay.pdf
├── references.bib
└── main.pdf
```

`archive/` contains the earlier, longer chapter drafts (not compiled; excluded
from the pushed copy).
