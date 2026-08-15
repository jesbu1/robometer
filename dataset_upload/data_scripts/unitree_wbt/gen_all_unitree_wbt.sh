#!/usr/bin/env bash

set -euo pipefail

DATASETS=(
  "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher"
  "G1_WBT_Brainco_Pickup_Pillow"
  "G1_WBT_Inspire_Put_Clothes_into_Washing_Machine"
  "G1_WBT_Brainco_Make_The_Bed"
  "G1_WBT_Inspire_Put_Clothes_Into_Basket"
)

for dataset_name in "${DATASETS[@]}"; do
  echo "Converting ${dataset_name}"
  uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/unitree_wbt.yaml \
    --dataset.dataset_name "${dataset_name}"
done
