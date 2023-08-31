output_file="examples/cover2.pdf"

compile:
	pandoc $(md_file) -o $(output_file) --template=template-cover.tex --pdf-engine=xelatex
	zathura $(output_file)
