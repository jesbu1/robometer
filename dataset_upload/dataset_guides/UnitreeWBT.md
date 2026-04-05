# UnitreeWBT Dataset Guide

This guide explains how to download, process, and upload Unitree WBT datasets with the Robometer dataset pipeline.

## Overview

Unitree WBT datasets are released in LeRobot format on Hugging Face. The loader converts each episode into one trajectory video and Hugging Face row for Robometer training.

- Source collection: [unitreerobotics/unifolm-wbt-dataset](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset)
- Camera policy during conversion: keep exactly one head stereo stream per trajectory, randomly sampled from:
  - `observation.images.head_stereo_left`
  - `observation.images.head_stereo_right`

## Prerequisites

1. Install HF CLI and authenticate:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
```

2. Install LeRobot in your conversion environment:

```bash
pip install -U lerobot
```

## Download all UnitreeWBT datasets

Run this command from the repo root to download all five datasets:

```bash
mkdir -p ./datasets/unitree_wbt && for repo in \
  unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  unitreerobotics/G1_WBT_Brainco_Pickup_Pillow \
  unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine \
  unitreerobotics/G1_WBT_Brainco_Make_The_Bed \
  unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket; do \
  name="${repo#*/}"; \
  hf download "$repo" --repo-type dataset --local-dir "./datasets/unitree_wbt/$name"; \
done
```

Direct links:

- https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher
- https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Pickup_Pillow
- https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine
- https://huggingface.co/datasets/unitreerobotics/G1_WBT_Brainco_Make_The_Bed
- https://huggingface.co/datasets/unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket

## Directory layout

Expected layout for conversion:

```text
./datasets/unitree_wbt/
  G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/
    meta/
    videos/
  G1_WBT_Brainco_Pickup_Pillow/
    meta/
    videos/
  ...
```

## Configuration

Use:

- `dataset_upload/configs/data_gen_configs/unitree_wbt.yaml`

Example:

```yaml
dataset:
  dataset_path: "./datasets/unitree_wbt"
  dataset_name: G1_WBT_Brainco_Collect_Plates_Into_Dishwasher

output:
  output_dir: ./robometer_dataset/unitree_wbt_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 1

hub:
  push_to_hub: true
  hub_repo_id: unitree_wbt_rfm
```

## Run conversion

Single dataset:

```bash
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path=dataset_upload/configs/data_gen_configs/unitree_wbt.yaml \
  --dataset.dataset_name G1_WBT_Brainco_Collect_Plates_Into_Dishwasher
```

All five datasets:

```bash
bash dataset_upload/data_scripts/unitree_wbt/gen_all_unitree_wbt.sh
```

## Notes

- If both stereo streams are present in an episode, the loader selects one view using deterministic random sampling per episode.
- If only one of left/right stereo streams exists, that stream is used.
- If neither stream exists, the episode is skipped.
