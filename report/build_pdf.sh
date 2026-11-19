#!/bin/bash
# ../REPORT.pdf = research note (pages 1-3) + detailed report + appendices; ../RESEARCH_NOTE.pdf = the note alone.
#   A: H1 tables (generated), B: figures, C: critique of the paper, D: the three pre-registrations.
# Needs pandoc + xelatex with DejaVu fonts. Run from anywhere.
set -e
cd "$(dirname "$0")"
# The note's figure: drop the "Figure 1." prefix (LaTeX numbers it) and set its width; pin floats in place.
sed -e 's#^!\[Figure 1: #![#' -e 's#](report/figures/fig_story.png)$#](report/figures/fig_story.png){width=72%}#' ../RESEARCH_NOTE.md > _note.md
cat > _figures.md <<'MD'
# Appendix B. Figures

![H1: every Swedish TEST Δ vs the unedited model. Shaded: the difference between two runs of the unedited model that differ only in sampling seed.](figures/fig_delta_sv.png){width=72%}

![H1: Swedish and English GSM8K accuracy (same 200 TEST problems) for each condition.](figures/fig_main_gsm.png){width=95%}

![H1: the DEV grid, every WSP-style edit (27 configurations).](figures/fig_sweep_dev.png){width=60%}

![H1 capability probe: en/ett agreement on DEV nouns vs edit strength. Amplifying the direction adds almost nothing; suppressing it removes the ability.](figures/fig_gender_dose.png){width=70%}

![H1: the paper's p50c edit and budget forcing, thinking on vs off.](figures/fig_expD_thinking.png){width=95%}

![H2: number-word edits vs the free alternative (numbers as digits), on the 59 TEST problems with number words.](figures/fig_numword.png){width=90%}
MD
{ cat _note.md
  printf '\n\\newpage\n\n'; cat report.md
  printf '\n\\newpage\n\n# Appendix A. H1 tables\n\n'; cat tables.md
  printf '\n\\newpage\n\n'; cat _figures.md
  printf '\n\\newpage\n\n'; sed 's/^# /# Appendix C. /' critique.md
  printf '\n\\newpage\n\n'; sed 's/^# /# Appendix D1. H1 — /' hypothesis_and_design.md
  printf '\n\\newpage\n\n'; sed -e 's/^# /# Appendix D2. H2 — /' -e 's/\\n/\\\\n/g' ../numword/preregistration.md
  printf '\n\n'; sed -e 's/^# /# Appendix D3. H3 — /' -e 's/\\n/\\\\n/g' ../swthink/preregistration.md
} > _report.md
OPTS=(-f markdown+lists_without_preceding_blankline --pdf-engine=xelatex --resource-path=.:.. \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V fontsize=10pt \
  -V geometry:margin=2cm -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -V header-includes='\usepackage{longtable}\setlength{\LTleft}{0pt}\usepackage{float}\floatplacement{figure}{H}')
pandoc "${OPTS[@]}" _report.md -o ../REPORT.pdf 2>&1 | grep -v "Missing character" | head -20
pandoc "${OPTS[@]}" _note.md -o ../RESEARCH_NOTE.pdf 2>&1 | grep -v "Missing character" | head -20
rm -f _report.md _figures.md _note.md
ls -la ../REPORT.pdf ../RESEARCH_NOTE.pdf
