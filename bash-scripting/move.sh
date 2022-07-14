#!/usr/bin/env sh


rename_file_extension () {
    # rename all files with the old extension to the new extension
    # ex: rename_file_extension "wrk" "py" will rename all files ending with 
    #     ".wrk" to ".py"
    old_extension=$1
    new_extension=$2
    for f in *.${old_extension}
    do 
	# echo "${f%.${old_extension}}.${new_extension}"
	mv  "$f" "${f%.${old_extension}}.${new_extension}"
    done
}

rename_file_extension "wrk" "py"
