#!/bin/bash
# run newnudenet2 on all files in a directory
# 
# usage: ./nudescript.sh
#
# requires: node, newnudenet2, and models/default-f16/model.json
#
# note: this script is not intended to be run as-is, but rather to be used as a
# reference for how to run newnudenet2 on a directory of images

# Set the directory containing the images
dir="./"

# Loop through all files in the directory
for file in "$dir"/*; do
    # Check if the file is a regular file
    if [ -f "$file" ]; then
        # Get image name
        filename=$(basename "$file")
        # Get extension
        extension="${filename##*.}"
        # Get filename without extension
        filename="${filename%.*}"
        # Set output path
        output="./here/${filename}_blurred.${extension}"
        # Run newnudenet2
        node src/newnudenet2.js \
          --model models/default-f16/model.json \
          --input "$file" \
          --output "$output" \
          --dir "$dir" \
          || echo "Error processing $file"
    fi
done
