#!/bin/bash
# use this script to sort and rename the cube files in a root directory
# usage: ./sort.sh <root dir>
# Change to the directory containing the .cube files
cd $1

# Initialize a counter
counter=1

# Loop over the files sorted alphabetically and rename them with a prefix indicating their rank
ls *.cube | sort | while IFS= read -r file; do
    filename=$(basename "$file")
    dir=$(dirname "$file")
    
    # Replace spaces with underscores and convert to lowercase
    new_filename=$(echo "$file" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    # Rename the file
    mv "$file" "$dir/${counter}-$new_filename"
    counter=$((counter+1))
done