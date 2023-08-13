# Cover Letter - Markdown to Latex


**Usage**

1. Write your cover letter inside `letter.md` and change the information in the YAML headers
2. Compile cover letter from markdown with makefile with the command: `make md_file=letter.md compile`
3. Cover File will be generated at `cover.pdf`


**Reminders**

* How to produce latex directly
    * `xelatex template-cover.tex`
* How to produce cover from markdown
    * `pandoc % -o %<.pdf --template=template-cover.tex`
    


## Ressources


[Tex CV Builder](https://github.com/antkr10/tex-cvbuilder/tree/main)
