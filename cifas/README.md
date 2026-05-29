# CIFAS 2026 Scrape

- https://event.fourwaves.com/fr/cifas2026/horaire?date=2026-06-16
- https://event.fourwaves.com/fr/cifas2026/resumes


> pandoc old_outline.md -o outline.pdf --pdf-engine=xelatex -V geometry:margin=2cm -V fontsize=11pt -V lang=fr -V mainfont="Arial Unicode MS"
> pandoc outputs/outline.md -o outline.pdf --pdf-engine=xelatex -V geometry:margin=2cm -V fontsize=11pt -V lang=fr -V mainfont="Arial Unicode MS"

1. Script qui genere la programmation en bref a partir de l'horaire https://event.fourwaves.com/fr/cifas2026/horaire?date=2026-06-16
2. Script qui obtient la description de tous les evenements https://event.fourwaves.com/fr/cifas2026/resumes  => `uv run python main.py`
3. Script qui ajoute la description a la programmation => `uv run python fill_outline.py` generated in `outputs/outline.md`
4. Convertir le fichier markdown a un fichier pdf avec latex et pandoc


## Reference Docs

- [arcAman07/TexGuardian](https://github.com/arcAman07/TexGuardian)
