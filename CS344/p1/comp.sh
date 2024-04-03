#!/bin/sh

# Author: yukikongju
# Description: This program compute mean/median from txt file based on rows (student grade) or columns (exam grade)


function mean {
    list=$1
    n=$( echo $list | wc -w )
    sum=$( echo $list | tr ' ' '\n' | awk '{sum+=$1} END {print sum}' )
    mean=$(( sum / n ))
    echo $mean
}

function median {
    list=$1
    n=$(echo $list | wc -w)
    sorted_list=$( echo $list | tr ' ' '\n' | sort -g | tr '\n' ' ' )

    # compute median
    if [ $((n % 2)) -eq 0 ]; then
	k=$((n/2))
	m0=$(echo "$sorted_list" | awk -v i="$k" '{print $i}')
	m1=$(echo "$sorted_list" | awk -v i="$((k+1))" '{print $i}')
	med=$(( (m0 + m1) / 2 ))
    else
	k=$((n/2 +1))
	med=$(echo "$sorted_list" | awk -v k="$k" '{print $k}')
    fi

    # return median
    echo $med
}

function read_grades_file {
    file_name=$1
    computation_method=$2
    grade_type=$3

    # --- computing results based on grade_type
    # results=()
    # case $grade_type in
    #     student*)
    #         while read line; do
    #             res=$( $computation_method "$line" )
    #         done < $file_name
    #         ;;
    #     exam*)
    #         num_cols=$(head -n 1 $file_name | wc -w)
    #         i=1
    #         while [ $i -le $num_cols ]; do
    #             col=$( cut -f $i $file_name )
    #             res=$( $computation_method "$col" )
    #             ((i++))
    #         done
    #     ;;
    # esac

    # echo $results


    # - reading row by row
    while read line; do
	res=$( $computation_method "$line" )
	echo $res
    done < $file_name
    
    echo

    # - reading col by col
    num_cols=$(head -n 1 $file_name | wc -w)
    i=1
    while [ $i -le $num_cols ]; do
	col=$( cut -f $i $file_name )
	res=$( $computation_method "$col" )
	echo $res
	((i++))
    done


}


function main {
    # --- Select Menu: (1) computation method: mean/median; (2) gradestype: student/exam
    computation_methods=("mean" "median")
    selected_computation_method=$(printf '%s\n' "${computation_methods[@]}" | fzf --reverse --with-nth 1 -d "\t" --header "Select a computation method:")

    # student: row-wise; exam: column-wise
    grade_types=("student" "exam")
    selected_grade_type=$(printf '%s\n' "${grade_types[@]}" |fzf --reverse --with-nth 1 -d "\t" --header "Select a grade type:")

    # --- Compute based on computation method and grade type
    case $selected_computation_method in
	mean*) echo "mean1" ;;
	median*) echo "median1" ;;
	*) echo "unavailable" ;;
    esac

}

read_grades_file "grades.txt" "mean" "student" 
# read_grades_file "grades.txt" "median" "student"
# main


