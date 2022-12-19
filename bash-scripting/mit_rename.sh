#!/usr/bin/bash

# script to rename MIT zip files
# ex file: 01645f86d111e8fdca0be5fd46920851_MIT18_05S14_Reading3.pdf
# https://stackoverflow.com/questions/4168371/how-can-i-remove-all-text-after-a-character-in-bash

# remove useless files
rm *.vtt *.srt

echo "Removing hash from pdf file name"
for file in $(ls *.pdf)
do 
    new_file=${file#*_}
    # rename MIT file if starts with hash and delete useless pdf files
    if [[ $new_file == *"MIT"* && ${file::1} != "M" ]]; then 
	mv "$file" "$new_file"
    elif [[ ${file::1} == "M" ]]; then
	continue
    else 
	rm "$file"
    fi
done

echo "create new directories to put pdf in"
slides_dir="Slides"
problems_dir="Problems"
reading_dir="Readings"
labs_dir="Labs"
exams_dir="Exams"
solutions_dir="Solutions"
recitations_dir="Recitations"
misc_dir="Miscellaneous"
homework_dir="Homework"
mkdir ${slides_dir} ${problems_dir} ${reading_dir} ${labs_dir} ${exams_dir} ${solutions} ${recitations_dir} ${misc_dir} ${homework_dir}

echo "moving file into their respective directory"
for file in $(ls *.pdf)
do 
    suffix=${file#*_}
    if [[ "${suffix}" == *"sol"* ]]; then
	mv ${file} "${recitations_dir}/${file}"
    elif [[ "${suffix}" == *"quiz"* ]]; then
	mv ${file} "${exams_dir}/${file}"
    elif [[ "${suffix}" == *"Problem"* ]]; then 
	mv ${file} "${problems_dir}/${file}"
    elif [[ "${suffix}" == *"hw"* ]]; then 
	mv ${file} "${homework_dir}/${file}"
    elif [[ "${suffix}" == *"Lecture"* ]]; then
	mv ${file} "${reading_dir}/${file}"
    else
	mv ${file} "${misc_dir}/${file}"
    fi
done

# remove useless directory
rmdir *

# move into files
course_dir="MIT 18.311 - Principles of Applied Maths"
mkdir "${course_dir}"

for dir in $(find . -type d)
do
    echo $dir
    mv "$dir" "${course_dir}"
done

# move into "More Courses"
more_course_dir="/home/yukikongju/Documents/OneDrive/Bac-Maths-Info/More Courses"
mv "${course_dir}" "${more_course_dir}"

