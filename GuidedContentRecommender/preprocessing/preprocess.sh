#!/bin/sh

MEDITATIONS_DIR=~/Data/Meditations
SLEEPTALES_DIR=~/Data/SleepTales

MEDITATIONS_OUTPUT_DIR=~/Data/Meditations_CLEAN
SLEEPTALES_OUTPUT_DIR=~/Data/SleepTales_CLEAN


convert_docx_to_txt() {
    directory_path=$1

    for docx_file in $directory_path/*.docx; do
	base_name=$(basename "$docx_file" .docx)
	output_file=$directory_path/$base_name.txt
	pandoc -s "$docx_file" -o "$output_file"
	echo "Succesfully converted $base_name"
    done

    # echo "Successfully converted word document inside $directory_path to $output_path"
}

remove_docx_files() {
    directory_path=$1
    rm $directory_path/*.docx
}

copy_txt_file_to_output_dir() { # DEPRECATED
    directory_path=$1
    output_path=$2

    for file in "$directory_path"/*.txt; do
	base_name=$(basename "$file" .txt)
	output_file=$output_path/$base_name.txt
	# cp $file $output_file
    done

}

get_invalid_EN_files() {
    ### EN files are the one without (*), _<LANG>, (LANG)
    directory_path=$1

    invalid_files=()
    substrings="FR SP RU JP PT ES é (\d)"
    for file in "$directory_path"/*; do
	# is_valid=true
	for substring in $substrings; do
	    if echo "$(basename "$file")" | grep -q "$substring"; then
		# echo "File $file contains substring $substring"
		# is_valid=false
		invalid_files+=("$file")
		break
	    fi
	done

	# adding file if valid
	# if [ "$is_valid" = true ]; then
	#     echo $file
	#     valid_files+=("$file")
	# fi
    done

    # echo ${valid_files[@]}
    echo ${invalid_files[@]}
}

rename_files_without_spacing_in_dir() {
    # replace space with underscore
    directory_path=$1

    for file in $directory_path/*; do
	new_name=$(echo "$(basename "$file")" | sed 's/ /_/g')
	new_file_path="$directory_path/$new_name"
	mv "$file" "$new_file_path"
    done
}


cleanup_meditations() {
    directory_path=$1
    output_path=$2

    # --- create/cleanup output directory if non-existent
    echo "--- Create/Cleanup Output directory ---\n"
    if [ ! -d $directory_path ]; then
	echo "Data directory doesn't exist. Please check!"
	exit 1
    fi

    if [ ! -d $output_path ]; then
	echo "Ouput path doesn't exist, creating..."
	mkdir $output_path
    else
	echo "Cleaning up existing directory"
	rm -rf $output_path
	mdkir $output_path
    fi


    # --- Copying files from raw to clean directory
    echo "--- Copy files from raw directory to clean directory ---\n"
    cp -r $directory_path $output_path

    # --- converting docx to txt
    echo "--- Convert docx to txt and removing old docx files ---\n"
    convert_docx_to_txt $output_path
    rm $output_path/*.docx

    # --- Filter for valid EN files
    echo "Filter for valid EN files"
    invalid_files=$(get_invalid_EN_files $output_path)
    rm $invalid_files

    # --- rename documents without spacing
    rename_files_without_spacing_in_dir $output_path

}


cleanup_sleeptales() {
    echo ""

}

# --- Cleaning up Meditations Files
cleanup_meditations $MEDITATIONS_DIR $MEDITATIONS_OUTPUT_DIR

# --- Cleaning up SleepTales Files

# filter_valid_EN_files $MEDITATIONS_DIR
# rename_files_without_spacing_in_dir $MEDITATIONS_DIR

# invalid_files=$(get_invalid_EN_files $MEDITATIONS_OUTPUT_DIR)
# rm $invalid_files
