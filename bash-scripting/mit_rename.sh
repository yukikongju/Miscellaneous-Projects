#!/bin/bash

# script to rename MIT zip files
# ex file: 01645f86d111e8fdca0be5fd46920851_MIT18_05S14_Reading3.pdf
# https://stackoverflow.com/questions/4168371/how-can-i-remove-all-text-after-a-character-in-bash

for file in $(ls *.pdf)
do 
    new_file=${file#*_}
    # echo "${file}"
    # echo "${new_file}"
    mv ${file} ${new_file}
done

echo "Renamed files pdf files!"
