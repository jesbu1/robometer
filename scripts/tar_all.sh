#!/bin/bash
set -euo pipefail

# Folders (relative to processed_datasets) to tar up. Add more as needed.
FOLDERS=(
    #"villekuosmanen_armnetbench_robometer_v01_so101"
    "ykorkmaz_abc_subset_rbm_abc_130k"
    "ykorkmaz_molmoact2_so100_101_rbm_molmoact2_so100_101"
    "ykorkmaz_usc_trossen_rfm_usc_trossen"
)

# Split threshold: folders larger than this get split into .tar.part-aa, .tar.part-ab, ...
SPLIT_THRESHOLD_BYTES=$((20 * 1024 * 1024 * 1024)) # 20 GiB
SPLIT_SIZE="20G"

# processed_datasets lives next to this script's parent (scripts/ -> ../processed_datasets)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESSED_DATASETS_DIR="$SCRIPT_DIR/../processed_datasets"
OUTPUT_DIR="$PROCESSED_DATASETS_DIR/large_folder_to_upload"

cd "$PROCESSED_DATASETS_DIR"
mkdir -p "$OUTPUT_DIR"

for folder in "${FOLDERS[@]}"; do
    if [ ! -d "$folder" ]; then
        echo "Skipping '$folder': not found in $PROCESSED_DATASETS_DIR"
        continue
    fi

    size_bytes=$(du -sb "$folder" | cut -f1)
    echo "Processing '$folder' ($size_bytes bytes)..."

    if [ "$size_bytes" -gt "$SPLIT_THRESHOLD_BYTES" ]; then
        echo "  Larger than threshold, tarring and splitting into $SPLIT_SIZE parts..."
        tar -cf - "$folder" | split -b "$SPLIT_SIZE" - "$OUTPUT_DIR/${folder}.tar.part-"
        echo "  Wrote $OUTPUT_DIR/${folder}.tar.part-*"
    else
        echo "  Tarring into a single archive..."
        tar -cf "$OUTPUT_DIR/${folder}.tar" "$folder"
        echo "  Wrote $OUTPUT_DIR/${folder}.tar"
    fi
done

echo "Done! Archives are in $OUTPUT_DIR"
