#!/usr/bin/env bash

set -euo pipefail

DEST_ROOT="./datasets/unitree_wbt"
mkdir -p "$DEST_ROOT"

REPOS=(
  "unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher"
  "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow"
  "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine"
  "unitreerobotics/G1_WBT_Brainco_Make_The_Bed"
  "unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket"
)

for repo in "${REPOS[@]}"; do
  name="${repo#*/}"
  out_dir="${DEST_ROOT}/${name}"
  echo "Downloading ${repo} -> ${out_dir}"
  hf download "$repo" --repo-type dataset --local-dir "$out_dir"
done

echo "All UnitreeWBT datasets downloaded to ${DEST_ROOT}."
