#!/bin/sh
# Author: yukikongju
# Description: This program compute mean/median from txt file based on rows (student grade) or columns (exam grade)


function read_grades_file {
    file_name=$1

    grades=()
    while IFS=$'\t' read -r -a line;do
	grades+=(${line[@]})
	echo ${line[@]}
    done < $file_name
}


function main {
    # --- Select Menu: Compute (1) mean/median
    computation_methods=("mean" "median")
    selected_computation_method=$(printf '%s\n' "${computation_methods[@]}" | fzf --reverse --with-nth 1 -d "\t" --header "Select a computation method:")


    # ---
    case $selected_computation_method in
	mean*) echo "mean1" ;;
	median*) echo "median1" ;;
	*) echo "unavailable" ;;
    esac
    

}

# read_grades_file "grades.txt"
main


