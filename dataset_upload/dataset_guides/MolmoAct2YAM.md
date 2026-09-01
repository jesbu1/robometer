# MolmoAct2 YAM Dataset Guide

This guide explains how to integrate and use the MolmoAct2 Bimanual YAM dataset with the Robometer training pipeline.

Source: `https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset`

## Overview

- **LeRobot v3.0 format** — single flat directory with `data/`, `meta/`, `videos/`.
- 32,246 episodes (trained sequences recorded at 30 FPS).
- Bimanual (YAM) robot manipulation trajectories with 14-dim state/action (6 joints + 1 gripper per arm).
- Three camera views per episode: `left`, `right`, `top`.
- Videos are AV1-encoded `.mp4` files with multiple episodes concatenated in a single file, accessed by timestamp.
- Per-episode task text stored in `meta/tasks_annotated.parquet`.

## Directory Structure

The dataset follows the standard LeRobot v3.0 layout under `<dataset_path>/`:

```
datasets/
  .gitattributes
  README.md
  data/
    chunk-000/
      file-000.parquet          # per-frame state/action data
      file-001.parquet
      ...
    chunk-001/
      ...
    chunk-002/
      ...
    chunk-003/
      ...
  meta/
    info.json                   # dataset metadata
    stats.json                  # normalization statistics
    tasks.parquet               # task_index -> task name (34 tasks)
    tasks_annotated.parquet     # episode_index -> task text
    episodes/
      chunk-000/
        file-000.parquet        # per-episode metadata (video ptrs, timestamps)
  videos/
    observation.images.left/
      chunk-000/
        file-000.mp4            # multiple episodes in one mp4, accessed via timestamps
        file-001.mp4
        ...
      chunk-001/
        ...
    observation.images.right/
      chunk-000/
        file-000.mp4
        ...
      chunk-001/
        ...
    observation.images.top/
      chunk-000/
        file-000.mp4
        ...
      chunk-001/
        ...
      chunk-002/
        ...
```

### Video → Episode Mapping

Each `.mp4` contains multiple concatenated episodes. The episodes parquet file maps each episode to its video segment via:

- `videos/observation.images.{camera}/chunk_index` — which chunk directory
- `videos/observation.images.{camera}/file_index` — which `.mp4` file within that chunk
- `videos/observation.images.{camera}/from_timestamp` — start position in seconds
- `videos/observation.images.{camera}/to_timestamp` — end position in seconds

### Data Parquet Fields

Each row in `data/chunk-*/file-*.parquet`:

| Column | Dtype | Description |
|---|---|---|
| `action` | list[float32] | 14-dim robot action |
| `observation.state` | list[float32] | 14-dim robot state |
| `timestamp` | float32 | Timestamp in seconds |
| `frame_index` | int64 | Per-episode frame index |
| `episode_index` | int64 | Episode identifier |
| `index` | int64 | Global frame index |
| `task_index` | int64 | Task type index |

## Configuration

```yaml
dataset:
  dataset_path: ./datasets              # Path to the downloaded LeRobot dataset
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
uv run hf download allenai/MolmoAct2-BimanualYAM-Dataset --local-dir=./datasets/ --repo-type=dataset
```

### Convert

```bash
uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/molmoact2_yam.yaml
```

### Convert (tiled)

Produces one trajectory per episode with all three views composited into a
single `640 x 540` frame — `top` in the primary upper panel, `left` and
`right` in the lower grid (see `dataset_upload/TILING.md`):

```bash
uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/molmoact2_yam_tiled.yaml
```

Output videos keep the untiled sampling conventions (32 frames at 10 fps) so
both variants stay comparable; only the image content differs. Tiled model
training/eval also works on raw multi-view inputs at inference time — the eval
server tiles views on the fly (see `dataset_upload/TILING.md`).

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Task description from task annotation
- `frames`: Relative path to the generated H.264 video clip
- `is_robot`: True
- `quality_label`: "successful"
- `data_source`: `molmoact2_yam` (or `molmoact2_yam_tiled` for the tiled variant)

## Notes

- Videos are originally AV1-encoded (all-keyframe); the converter decodes and re-encodes to H.264 via ffmpeg.
- Untiled converter: only the `top` camera view is extracted per episode (produces one trajectory per episode).
- Tiled converter: all three views are decoded per episode (each camera has its own video pointer/timestamps) and composited per frame.
- Frames are extracted on-demand via ffmpeg piping for memory efficiency.
- 32,246 total episodes across 34 unique tasks.
- Training dataset category: `rbm-1.1-tiled-molmoact2`; preprocessed name: `jesbu1_molmoact2_yam_tiled_rfm_molmoact2_yam_tiled` (short name `molmoact2_yam_tiled`, success cutoff 0.94).
