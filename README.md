# Privacy Failure in Split-LLM Training: The Returned Gradient Nullifies the Decoys

George Politis, Evangelos Pappas — arXiv technical report, August 2026.

## Repository layout

- `main.tex` — paper source (LaTeX, `article` class, 11pt)
- `refs.bib` — bibliography (BibTeX)
- `main.bbl` — pre-built bibliography for arXiv submission (generated; not needed for local builds)
- `figs/` — figures included by the paper (`fig_w24_dose.pdf`, `fig_w56_shape.pdf`)
- `placeins.sty` — local copy of the `placeins` package (already in TeX Live; kept for arXiv portability)
- `anc/` — ancillary material: attack code, experiment scripts, frozen data, and provenance records (see `anc/README.md`)

## Building the PDF

You need a TeX distribution (TeX Live / MacTeX) with `pdflatex`, `bibtex`, and the standard packages (`tikz`, `booktabs`, `microtype`, etc.).

From the repository root:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or with `latexmk` (handles the passes automatically):

```bash
latexmk -pdf main.tex
```

The output is `main.pdf`. All figures are already committed under `figs/`, so no data regeneration is required to build the paper.

## arXiv submission

Submit `main.tex`, `main.bbl`, and `figs/`. arXiv compiles with the shipped `.bbl`, so `refs.bib` and `placeins.sty` are optional.
