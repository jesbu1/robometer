# Tiled Views

This repository supports converting synchronized camera views into one image or
video frame. The reusable implementation is:

```text
dataset_upload/helpers.py:tile_synchronized_views
```

## Layout

The utility creates a large primary-view panel and a grid of secondary views.
By default, the primary panel uses two thirds of the output height. Secondary
views are placed in an approximately square grid in the remaining space.

For three views, the result is:

```text
+---------------------------+
|                           |
|       primary view        |
|                           |
+-------------+-------------+
| secondary 1 | secondary 2 |
+-------------+-------------+
```

Unused grid cells are filled with the configured background value.

## Python API

`tile_synchronized_views` accepts a mapping of view names to RGB arrays with
shape `(T, H, W, 3)`:

```python
from dataset_upload.helpers import tile_synchronized_views

tiled = tile_synchronized_views(
    frames_by_view={
        "top": top_frames,
        "left": left_frames,
        "right": right_frames,
    },
    primary_view="top",
    secondary_views=["left", "right"],
    target_width=640,
    target_height=540,
)
```

The function truncates all views to the shortest synchronized frame count,
resizes each panel with OpenCV area interpolation, and returns a `uint8` RGB
array. The order in `secondary_views` determines the lower-grid placement.

The number of secondary views is not fixed. For example, four secondary views
are automatically arranged as a 2x2 lower grid. Callers should provide views
that are already synchronized by timestamp or frame index.

## Armnet

The native Armnet LeRobot converter is:

```text
dataset_upload/dataset_loaders/armnet_lerobot_loader.py
```

It uses these layouts:

```text
Single-arm: top, front, wrist
Bimanual:   top, left_wrist, right_wrist
```

The current generated datasets use `640 x 540` frames. The raw conversion
configs are:

```text
dataset_upload/configs/armnet_tiled_so101.yaml
dataset_upload/configs/armnet_tiled_bimanual_so101.yaml
```

The converter preserves the native episode labels:

```text
successful
failure
suboptimal
```

`failure` and `suboptimal` remain distinct in the processed metadata. The RBM
training code treats both as non-successful trajectories for progress masking,
success labels, and preference sampling.

## MolmoAct2 YAM

The bimanual YAM dataset is tiled with the same utility by:

```text
dataset_upload/dataset_loaders/molmoact2_yam_loader.py:convert_molmoact2_yam_tiled_to_hf
```

Layout: `top` is the primary panel, `left` and `right` share the lower grid.
Each camera has independent LeRobot video pointers and timestamps, so views are
synchronized per camera by decoding each camera's own segment. The conversion
config is:

```text
dataset_upload/configs/data_gen_configs/molmoact2_yam_tiled.yaml
```

Output is `640 x 540`, 32 frames at 10 fps per trajectory, matching the
untiled `molmoact2_yam_rfm` sampling conventions. All episodes carry the
native `successful` label.

## Dynamic Tiling at Inference

The tiled checkpoints can be hosted and served with either input style:

```text
1. Pre-tiled videos    — a single image stream, sent as `frames`.
2. Raw multi-view data — a `views` mapping of view name to (T, H, W, 3) arrays.
```

For raw multi-view inputs, the eval server tiles the views on the fly with
`robometer/utils/tiling.py:apply_dynamic_tiling_to_trajectory` using the same
layout as the tiled training data (`primary_view` upper panel, ordered
`secondary_views` in the lower grid, canvas size inferred from the primary
view's aspect when not given). The client never needs the robometer package:

```bash
python scripts/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --view top=camera_top.mp4 --view left=camera_left.mp4 --view right=camera_right.mp4 \
  --primary-view top \
  --task "Move the blocks to spell AI2"
```

`scripts/example_inference_local.py` supports the same `--view` flags and
tiles locally before inference. Raw multi-view dicts with a `views` key are
also accepted by `robometer/evals/eval_utils.py:raw_dict_to_sample` for
rollout-style wrappers.

## Preprocessing and Training

The tiled preprocessing configs are:

```text
robometer/configs/preprocess_rbm1.1_tiled_armnet.yaml
robometer/configs/preprocess_rbm1.1_tiled_molmoact2.yaml
```

The tiled fine-tuning launchers explicitly set the tiled image resolution:

```text
run_scripts/finetune_armnet_tiled_qwen3_fft.sbatch
run_scripts/finetune_molmoact2_tiled_qwen3_fft.sbatch
```

The resolution is not a global default. Other dataset converters and training
launchers retain their existing image settings unless they explicitly opt into
the tiled configuration.

## Uploaded Artifacts

Datasets:

- `jesbu1/armnetbench_v01_tiled_so101`
- `jesbu1/armnetbench_v01_tiled_bimanual_so101`
- `jesbu1/molmoact2_yam_tiled_rfm`

Model:

- `jesbu1/robometer-4b-fft-armnet-tiled`
