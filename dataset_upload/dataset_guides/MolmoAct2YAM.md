# MolmoAct2 YAM Dataset Guide

This guide explains how to integrate and use the MolmoAct2 Bimanual YAM dataset with the Robometer training pipeline.

Source: `https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset`

## Overview

- LeRobot/Parquet datasets with bimanual (YAM) robot manipulation trajectories.
- Three camera views per episode: `left`, `right`, `top`.
- Pre-recorded AV1 video files with per-episode timestamp ranges.
- Task text per episode stored in `meta/tasks_annotated.parquet`.

## Directory Structure

Each cached dataset follows the HuggingFace datasets cache format under `<dataset_path>/`:

```
datasets--allenai--{date}-yam-{id}/
  snapshots/
    {hash}/
      data/
        chunk-000/
          file-*.parquet          # state/action data (no images)
      meta/
        info.json                 # dataset metadata
        tasks.parquet             # task_index -> task name
        tasks_annotated.parquet   # episode_index -> task text
        episodes/
          chunk-000/
            file-000.parquet      # per-episode metadata (video ptrs, timestamps)
      videos/
        observation.images.left/
          chunk-000/
            file-*.mp4
        observation.images.right/
          chunk-000/
            file-*.mp4
        observation.images.top/
          chunk-000/
            file-*.mp4
```

## Configuration

```yaml
dataset:
  dataset_path: /data/molmoact2_data       # Cache dir with all YAM subdirs
  dataset_name: molmoact2_yam

output:
  output_dir: ./robometer_dataset/molmoact2_yam_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 1

hub:
  push_to_hub: true
  hub_repo_id: molmoact2_yam_rfm
```

## Usage

### Download

```bash
HF_TOKEN=hf_xxx uv run python dataset_upload/data_scripts/molmoact2_yam/download_molmoact2_yam.py
```

### Convert

```bash
uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/molmoact2_yam.yaml
```

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Task description from task annotation
- `frames`: Relative path to the generated H.264 video clip
- `is_robot`: True
- `quality_label`: "successful"
- `data_source`: `molmoact2_yam`

## Notes

- Videos are originally AV1-encoded; the converter decodes and re-encodes to H.264.
- Each episode produces three trajectories (one per camera view: left, right, top).
- Frames are extracted on-demand via ffmpeg piping for memory efficiency.
- 15 YAM sub-datasets with ~600 total episodes.
