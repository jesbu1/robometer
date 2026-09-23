# RoboCasa365 Dataset Guide

This guide explains how to integrate and use the RoboCasa365 pretraining datasets with the Robometer training pipeline.

Source: `https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-atomic` and `https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-composite` (flat LeRobot v3.0 mirrors of the RoboCasa365 human pretraining corpora).

## Overview

- **LeRobot v3.0 format** — one source tree per subset with `data/`, `meta/`, `videos/`.
- 7,356 atomic episodes + 24,687 composite episodes (human teleoperation, recorded at 20 FPS).
- RoboCasa365 kitchen manipulation trajectories (Panda + Omron robots).
- Three camera views per episode: `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`.
- Videos are H.264-encoded `.mp4` files with multiple episodes concatenated in a single file, accessed by timestamp.
- Per-episode task text stored in the `tasks` column of `meta/episodes/*.parquet`.

## Directory Structure

The dataset follows the standard LeRobot v3.0 layout under `<dataset_path>/`:

```
datasets/
  robocasa365-pretrain-atomic/
    data/
      chunk-000/
        file-000.parquet          # per-frame state/action data
        ...
    meta/
      info.json                   # dataset metadata (fps, shapes, paths)
      stats.json                  # normalization statistics
      tasks.parquet               # task_index -> task text
      episodes/
        chunk-000/
          file-000.parquet        # per-episode metadata incl. camera pointers + tasks
    videos/
      observation.images.robot0_agentview_left/
        chunk-000/
          file-000.mp4
      observation.images.robot0_agentview_right/
        chunk-000/
          file-000.mp4
      observation.images.robot0_eye_in_hand/
        chunk-000/
          file-000.mp4
  robocasa365-pretrain-composite/
    ...
```

## Conversion

The tiled converter is:

```text
dataset_upload/dataset_loaders/robocasa365_loader.py:convert_robocasa365_tiled_to_hf
```

with config:

```text
dataset_upload/configs/data_gen_configs/robocasa365_tiled.yaml
```

Run with:

```bash
uv run python dataset_upload/generate_hf_dataset.py --config dataset_upload/configs/data_gen_configs/robocasa365_tiled.yaml
```

- `dataset_path` accepts a comma-separated list of source trees; sources are
  processed in order and the trajectory index continues across them.
- `left` (agentview_left) is the primary upper panel; `right` (agentview_right)
  and `wrist` (eye_in_hand) share the lower grid.
- Output is `512 x 768`, 32 frames at 10 fps per trajectory — primary panel
  `512 x 512`, secondary cells `256 x 256`, so the square 256x256 source views
  keep their aspect ratio everywhere.
- All episodes carry the `successful` label; `data_source` is `robocasa365_tiled`.

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Task description from the episode `tasks` column
- `frames`: Relative path to the generated H.264 video clip
- `is_robot`: True
- `quality_label`: "successful"
- `data_source`: `robocasa365_tiled`

## Notes

- Both atomic and composite subsets are merged into a single output dataset; the trajectory index continues across sources.
- Each camera has its own video pointer/timestamps; views are synchronized per camera by decoding each camera's own segment.
- Frames are extracted on-demand via ffmpeg piping for memory efficiency.
- 32,043 total episodes across ~5,045 unique task phrasings (617 atomic + 4,428 composite).
- Training dataset category: `rbm-1.1-tiled-robocasa365`; preprocessed name: `jesbu1_robocasa365_tiled_rfm_robocasa365_tiled` (short name `robocasa365_tiled`, success cutoff 0.95).
- Preprocessing config: `robometer/configs/preprocess_rbm1.1_tiled_robocasa.yaml`.
- The processed cache is published to `jesbu1/processed_dataset_rbm1.1_tiled` as split tars; extract with `cat <folder>.tar.part-* | tar -xvf -` inside `$ROBOMETER_PROCESSED_DATASETS_PATH` (see `scripts/tar_and_upload_processed.sh`).
- Tiled model training/eval also works on raw multi-view inputs at inference time — the eval server tiles views on the fly (see `dataset_upload/TILING.md`).