#!/usr/bin/env bash
# ============================================================
#  convert_cifas2026.sh
#  Convertit le fichier Markdown du CIFAS 2026 en PDF via Pandoc + XeLaTeX
#
#  Usage :
#    chmod +x convert_cifas2026.sh
#    ./convert_cifas2026.sh
#
#  Prérequis :
#    - pandoc >= 2.19
#    - xelatex (TeX Live ou MiKTeX)
#    - polices : TeX Gyre Termes / Heros (incluses dans TeX Live)
# ============================================================

# INPUT="Programmation_scientifique_CIFAS_2026.txt"   # votre fichier Markdown
# INPUT="cifas2026_presentations3.md"   # votre fichier Markdown
INPUT="outputs/outline.md"   # votre fichier Markdown
OUTPUT="cifas2026_programme_pandoc.pdf"
TEMPLATE="${1:-template_cifas_v2.tex}"     # template Pandoc (v2 = style sobre, v1 = template_cifas.tex)

pandoc \
  "$INPUT" \
  --output "$OUTPUT" \
  --pdf-engine=xelatex \
  --template="$TEMPLATE" \
  --toc \
  --toc-depth=2 \
  --number-sections \
  --syntax-highlighting=pygments \
  --variable mainfont="Arial Unicode MS" \
  --variable sansfont="Arial Unicode MS" \
  --variable fontsize=10pt \
  --variable papersize=a4 \
  --variable geometry="top=18mm,bottom=18mm,inner=18mm,outer=14mm" \
  --variable lang=french \
  --variable colorlinks=true \
  --variable linkcolor="0,82,155" \
  --variable documentclass=article \
  --metadata title="X\textsuperscript{e} CIFAS 2026 — Programme scientifique" \
  --metadata author="Comité organisateur du CIFAS 2026" \
  --metadata date="Montréal, 15–18 juin 2026" \
  && echo "✅  PDF généré : $OUTPUT" \
  || echo "❌  Erreur lors de la conversion — vérifiez les logs ci-dessus."
