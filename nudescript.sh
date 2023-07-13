for file in /mnt/media/storage3/PICARCHIVE/2023-Q3/2023-07/text/*.png; do
    filename=$(basename "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    output=./here/${filename}_blurred.${extension}
    node src/newnudenet2.js --model models/default-f16/model.json --input "$file" --output "$output"
done

