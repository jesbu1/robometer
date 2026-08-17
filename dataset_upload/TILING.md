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

## Preprocessing and Training

The tiled preprocessing config is:

```text
robometer/configs/preprocess_rbm1.1_tiled_armnet.yaml
```

The tiled fine-tuning launcher explicitly sets the tiled image resolution:

```text
run_scripts/finetune_armnet_tiled_qwen3_fft.sbatch
```

The resolution is not a global default. Other dataset converters and training
launchers retain their existing image settings unless they explicitly opt into
the tiled configuration.

## Uploaded Artifacts

Datasets:

- `jesbu1/armnetbench_v01_tiled_so101`
- `jesbu1/armnetbench_v01_tiled_bimanual_so101`

Model:

- `jesbu1/robometer-4b-fft-armnet-tiled`
