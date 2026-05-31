#!/usr/bin/env bash
# ============================================================
#  convert.sh  —  CIFAS 2026 Markdown → PDF
#
#  Usage :
#    bash convert.sh                   # template v2 (défaut)
#    bash convert.sh template_cifas.tex   # template v1
# ============================================================

# INPUT="outputs/outline.md"
# INPUT="outputs/outline_test5.md"
INPUT="outputs/outline_final.md"
OUTPUT="tests/demo.pdf"
TEMPLATE="${1:-template_cifas_v2.tex}"

  # -- toc \
  # --toc-depth=3 \

pandoc "$INPUT" \
  --output "$OUTPUT" \
  --pdf-engine=lualatex \
  --template="$TEMPLATE" \
  --lua-filter=cifas_headings_bouba.lua \
  --variable pagestyle=empty \
  --variable mainfont="Avenir Next" \
  --variable sansfont="Avenir Next" \
  --variable monofont="Courier New" \
  --variable fontsize=11pt \
  --variable papersize=a4 \
  --variable lang=french \
  --variable colorlinks=true \
  --metadata title="X\textsuperscript{e} CIFAS 2026 — Programme scientifique" \
  --metadata author="Comité organisateur du CIFAS 2026" \
  --metadata date="Montréal, 15–18 juin 2026" \
  && echo "✅  PDF généré : $OUTPUT" \
  || echo "❌  Erreur — vérifiez les logs ci-dessus."
