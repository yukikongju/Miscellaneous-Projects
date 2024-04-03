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

function compute_results {
    file_name=$1
    computation_method=$2
    grade_type=$3

    # --- computing results based on grade_type
    results=()
    case $grade_type in
	student*) 
	    # - reading row by row
	    while read line; do
		res=$( $computation_method "$line" )
		# echo $line
		results+=($res)
	    done < $file_name
	    ;;
	exam*)
	    # - reading col by col
	    num_cols=$(head -n 1 $file_name | wc -w)
	    i=1
	    while [ $i -le $num_cols ]; do
		col=$( cut -f $i $file_name )
		res=$( $computation_method "$col" )
		# echo $col
		results+=($res)
		((i++))
	    done
	;;
    esac

    # echo
    echo ${results[@]}

}

function usage {
    printf "
	Description:
	This program compute mean/median from txt file based on rows (student grade) or columns (exam grade)

	Usage: %s [options] [query]
	If a query is provided, it will be used to compute the grades
	
	Options:
	  -f, --file
	    Specify grades files
	  -m, --method
	    Specify computation method. Values: 'mean', 'median'
	  -g, --grade
	    Specify grade type. Values: 'student' (row-wise), 'exam' (column-wise)
	  -v, --verbose
	    Show additional message
    "
}


function main {
    # --- Select Menu: (1) computation method: mean/median; (2) gradestype: student/exam
    computation_methods=("mean" "median")
    selected_computation_method=$(printf '%s\n' "${computation_methods[@]}" | fzf --reverse --with-nth 1 -d "\t" --header "Select a computation method:")

    # student: row-wise; exam: column-wise
    grade_types=("student" "exam")
    selected_grade_type=$(printf '%s\n' "${grade_types[@]}" |fzf --reverse --with-nth 1 -d "\t" --header "Select a grade type:")

    # --- Compute based on computation method and grade type
    compute_results $file_name $selected_computation_method $selected_grade_type

}

# --- test cases
# compute_results "grades.txt" "mean" "student" # okay
# compute_results "grades.txt" "mean" "exam" # okay
# compute_results "grades.txt" "median" "student" # okay
# compute_results "grades.txt" "median" "exam" # okay

file_name="grades.txt"
# main
usage


