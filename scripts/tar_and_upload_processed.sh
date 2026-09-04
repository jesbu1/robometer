#!/bin/bash
# Tar processed-dataset caches into <=20GB parts and stream-upload each part to
# HuggingFace without materializing the full archive (works on machines without
# enough free disk for the whole tar). Part naming matches scripts/tar_all.sh:
#   <folder>.tar.part-aa, .tar.part-ab, ...
#
# Usage:
#   scripts/tar_and_upload_processed.sh <folder> <repo_id>
# Example:
#   scripts/tar_and_upload_processed.sh jesbu1_molmoact2_yam_tiled_rfm_molmoact2_yam_tiled jesbu1/processed_dataset_rbm1.1_tiled
#
# Extract on the cluster with:
#   cat <folder>.tar.part-* | tar -xvf -   (run inside $ROBOMETER_PROCESSED_DATASETS_PATH)
set -euo pipefail

FOLDER=${1:?folder inside processed_datasets required}
REPO_ID=${2:?HF dataset repo id required}

SPLIT_SIZE="20G"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESSED_DATASETS_DIR="$SCRIPT_DIR/../processed_datasets"
OUTPUT_DIR="$PROCESSED_DATASETS_DIR/large_folder_to_upload"

cd "$PROCESSED_DATASETS_DIR"
if [ ! -d "$FOLDER" ]; then
    echo "Folder not found: $PROCESSED_DATASETS_DIR/$FOLDER" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

echo "Streaming tar of '$FOLDER' (20GB parts) to $REPO_ID ..."
tar -cf - "$FOLDER" \
  | split -b "$SPLIT_SIZE" --filter="$SCRIPT_DIR/upload_tar_part.py '$REPO_ID'" - "$OUTPUT_DIR/${FOLDER}.tar.part-"

rm -f "$OUTPUT_DIR/${FOLDER}.tar.part-"*
echo "All parts uploaded to $REPO_ID"
