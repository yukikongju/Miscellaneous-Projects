cleanup:
	rm *.snm *.toc *.vrb *.out *.nav *.log *.aux

compile:
	# xelatex beamerthemeNord.tex
	# make cleanup
	pandoc test.md -o test.pdf --pdf-engine=xelatex --template=beamerthemeNord.tex 
	# pandoc test.md -o test.pdf --pdf-engine=xelatex --template=beamerthemeNord.tex -t beamer


