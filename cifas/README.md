# CIFAS 2026 Scrape

- https://event.fourwaves.com/fr/cifas2026/horaire?date=2026-06-16
- https://event.fourwaves.com/fr/cifas2026/resumes


> pandoc old_outline.md -o outline.pdf --pdf-engine=xelatex -V geometry:margin=2cm -V fontsize=11pt -V lang=fr -V mainfont="Arial Unicode MS"
> pandoc outputs/outline.md -o outline.pdf --pdf-engine=xelatex -V geometry:margin=2cm -V fontsize=11pt -V lang=fr -V mainfont="Arial Unicode MS"
> pandoc tests/demo.md --include-in-header fancyheaders.tex --pdf-engine=lualatex --toc -o tests/demo.pdf
> pandoc -N --variable "geometry=margin=1.2in" --variable mainfont="Palatino" --variable sansfont="Helvetica" --variable monofont="Menlo" --variable fontsize=12pt tests/demo.md --pdf-engine=lualatex --toc -o tests/demo.pdf
> pandoc --variable "geometry=margin=1.2in" --variable mainfont="Palatino" --variable sansfont="Palatino" --variable monofont="Palatino" --variable fontsize=12pt tests/demo.md --pdf-engine=lualatex --toc -o tests/demo.pdf

>   pandoc --variable "geometry=margin=1.2in" \
    --variable mainfont="Palatino" \
    --lua-filter cifas_headings.lua \
    --pdf-engine=lualatex --toc \
    outputs/outline.md -o tests/demo.pdf

> pandoc --variable "geometry=margin=1.2in" \
    --variable mainfont="Palatino" \
    --variable sansfont="Palatino" \
    --variable monofont="Palatino" \
    --lua-filter cifas_headings.lua \
    --pdf-engine=lualatex --toc \
    outputs/outline_final.md -o tests/demo.pdf

> pandoc --variable "geometry=margin=1.2in" \
    --variable mainfont="Avenir Next" \
    --variable sansfont="Avenir Next" \
    --variable monofont="Courier New" \
    --variable pagestyle=empty \
    --lua-filter cifas_headings_bouba.lua \
    --pdf-engine=lualatex \
    outputs/outline_final_presentation.md -o tests/demo.pdf

> pandoc --variable "geometry=margin=1.2in" \
    --variable mainfont="Avenir Next" \
    --variable sansfont="Avenir Next" \
    --variable monofont="Courier New" \
    --variable pagestyle=empty \
    --lua-filter cifas_headings_bouba.lua \
    --pdf-engine=lualatex \
    outputs/outline_final_presentation.md -o tests/demo.docx


----

> pandoc --variable "geometry=margin=1.2in" --variable mainfont="Palatino" --variable sansfont="Palatino" --variable monofont="Palatino" --variable fontsize=12pt tests/demo.md --pdf-engine=lualatex --lua-filter filter.lua --toc -o tests/demo.pdf

1. Script qui genere la programmation en bref a partir de l'horaire https://event.fourwaves.com/fr/cifas2026/horaire?date=2026-06-16
2. Script qui obtient la description de tous les evenements https://event.fourwaves.com/fr/cifas2026/resumes  => `uv run python main.py`
3. Script qui ajoute la description a la programmation => `uv run python fill_outline.py` generated in `outputs/outline.md`
4. Convertir le fichier markdown a un fichier pdf avec latex et pandoc


```
uv run python main.py
uv run python fill_outline.py
```


## Reference Docs

- [arcAman07/TexGuardian](https://github.com/arcAman07/TexGuardian)
