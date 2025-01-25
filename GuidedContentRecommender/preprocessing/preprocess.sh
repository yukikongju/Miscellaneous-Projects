#!/bin/sh

MEDITATIONS_DIR=~/Data/Meditations
SLEEPTALES_DIR=~/Data/SleepTales

MEDITATIONS_OUTPUT_DIR=~/Data/Meditations_CLEAN
SLEEPTALES_OUTPUT_DIR=~/Data/SleepTales_CLEAN


convert_docx_to_txt() {
    directory_path=$1

    for docx_file in $directory_path/*.docx; do
	base_name=$(basename $docx_file .docx)
	output_file=$directory_path/$base_name.txt
	echo $output_file
	# pandoc -s $docx_file -o $output_file
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
	base_name=$(basename $file .txt)
	output_file=$output_path/$base_name.txt
	# cp $file $output_file
    done

}

filter_valid_EN_files() {
    ### EN files are the one without (*), _<LANG>, (LANG)
    directory_path=$1

    valid_files=()
    substrings="FR SP RU JP PT ES é (\d)"
    for file in $directory_path/*; do
	is_valid=true
	for substring in $substrings; do
	    if echo "$(basename "$file")" | grep -q "$substring"; then
		# echo "File $file contains substring $substring"
		is_valid=false
		break
	    fi
	done

	# adding file if valid
	if [ "$is_valid" = true ]; then
	    echo $file
	    valid_files+=($file)
	fi
    done

    echo ${valid_files[@]}
}

rename_file_without_spacing() {
    # replace space with underscore
    file_name=$1
    new_name=" echo $(basename "$file_name") | sed 's/ /_/g'"
}


cleanup_meditations() {
    $directory_path=$1
    $output_path=$2

    # --- create/cleanup output directory if non-existent
    if [ ! -d $directory_path ]; then
	echo "Data directory doesn't exist. Please check!"
	exit 1
    fi

    if [ ! -d $output_path ]; then
	echo "Ouput path doesn't exist, creating..."
	mkdir $output_path
    else
	echo "Cleaning up existing directory"
	# TODO: remove files in directory?
    fi

    # --- Copying files from raw to clean directory
    echo "Copy files from raw directory to clean directory"
    cp -r $directory_path $output_path

    # --- converting docx to txt
    echo "convert docx to txt and removing old docx files"
    convert_docx_to_txt $output_path
    rm $output_path/*.docx

    # --- Filter for valid EN files
    valid_files=$(filter_valid_EN_files $MEDITATIONS_DIR)

    # --- TODO: rename documents without spacing


}


cleanup_sleeptales() {
    echo ""

}

# --- Cleaning up Meditations Files
# cleanup_meditations $MEDITATIONS_DIR $MEDITATIONS_OUTPUT_DIR

# --- Cleaning up SleepTales Files

# filter_valid_EN_files $MEDITATIONS_DIR
