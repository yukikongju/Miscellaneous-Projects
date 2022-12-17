#!/bin/bash

# script to rename MIT zip files
# ex file: 01645f86d111e8fdca0be5fd46920851_MIT18_05S14_Reading3.pdf
# https://stackoverflow.com/questions/4168371/how-can-i-remove-all-text-after-a-character-in-bash

echo "Removing hash from pdf file name"
for file in $(ls *.pdf)
do 
    new_file=${file#*_}
    mv ${file} ${new_file}
done

echo "create new directories to put pdf in"
slides_dir="Slides"
problems_dir="Problems"
reading_dir="Readings"
labs_dir="Labs"
exams_dir="Exams"
mkdir ${slides_dir} ${problems_dir} ${reading_dir} ${labs_dir} ${exams_dir}

echo "moving file into their respective directory"
for file in $(ls *.pdf)
do 
    suffix=${file#*_}
    if [[ "${suffix}" == *"ps"* ]]; then 
	mv ${file} "${problems_dir}/${file}"
    elif [[ "${suffix}" == *"Read"* ]]; then
	mv ${file} "${reading_dir}/${file}"
    elif [[ "${suffix}" == *"studio"* ]]; then
	mv ${file} "${labs_dir}/${file}"
    elif [[ "${suffix}" == *"slides"* ]]; then
	mv ${file} "${slides_dir}/${file}"
    elif [[ "${suffix}" == *"Exa"* ]]; then
	mv ${file} "${exams_dir}/${file}"
    fi
done


