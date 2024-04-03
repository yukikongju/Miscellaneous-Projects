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


file_name="grades.txt"
# usage

# --- process flags
n=$# # number of parameters
args=("$@") # store args in array
i=0
while [ $i -lt $n ]; do
    param=${args[$i]}
    case $param in
	-f | --file) # read file_name
	    if [ $((i+1)) -lt $n ]; then
		file_name="${args[i+1]}"
		((i++))
	    else
		echo "Usage: $0 -f <FILE_NAME>.txt"
	    fi
	    ;;
	-g | --grade) # select grade type: "student" or "exam"
	    if [ $((i+1)) -le $n ]; then
		selected_grade_type="${args[i+1]}"
		((i++))
	    else
		echo "Usage: $0 -g <GRADE>. Values: 'student', 'exam' "
	    fi
	    ;;
	-m | --method) # select computation method: "mean" or "median"
	    if [ $((i+1)) -le $n ]; then
		selected_computation_method="${args[i+1]}"
		((i++))
	    else
		echo "Usage: $0 -m <METHOD>. Values: 'mean', 'median' "
	    fi
	    ;;
	-v | --verbose)
	    # set -x
	    verbose=1
	    ;;
	*) echo "Unrecognized flag";;
    esac

    ((i++))
done

# --- check if variables are set. if not, then ask with menu
if [ -z "$selected_computation_method" ]; then
    computation_methods=("mean" "median")
    selected_computation_method=$(printf '%s\n' "${computation_methods[@]}" | fzf --reverse --with-nth 1 -d "\t" --header "Select a computation method:")
fi

if [ -z "$selected_grade_type" ]; then
    grade_types=("student" "exam")
    selected_grade_type=$(printf '%s\n' "${grade_types[@]}" |fzf --reverse --with-nth 1 -d "\t" --header "Select a grade type:")
fi

# --- Compute based on computation method and grade type
compute_results $file_name $selected_computation_method $selected_grade_type

# ---------------------------- TESTS --------------------

# --- test cases: set 1
# compute_results "grades.txt" "mean" "student" # okay
# compute_results "grades.txt" "mean" "exam" # okay
# compute_results "grades.txt" "median" "student" # okay
# compute_results "grades.txt" "median" "exam" # okay

# --- test cases: set 2
# ./comp.sh -g "student" -m "mean" # okay
# ./comp.sh -g "student" -m "median" # okay
# ./comp.sh -g "exam" -m "mean"  # okay
# ./comp.sh -g "exam" -m "median" # okay
