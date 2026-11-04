# Tiled Reward Evaluation Server

Serve a Robometer checkpoint over HTTP so you can send **raw camera videos** and get
per-frame rewards back. The server decodes the videos, synchronizes the views,
tiles them into the single-image layout the checkpoint was trained on, and runs
inference — no client-side preprocessing needed.

The tiling implementation is shared with training:
`robometer/utils/tiling.py:tile_synchronized_views`.

## Quickstart

### 1. Start the server

```bash
sbatch run_scripts/serve_tiled_eval.sbatch \
  /path/to/checkpoint 8010
```

Both arguments are optional:
- `MODEL_PATH` defaults to the busybox tiled checkpoint
  (`logs/rbm_fft_busybox_tiled_qwen3_4b_16bs_4gpu/ckpt-avg-2metrics=0.9219_step=750`)
- `PORT` defaults to `8010`

The job prints its node and the exact URL to use. Direct invocation (inside an
existing GPU allocation) also works:

```bash
uv run python robometer/evals/eval_server.py \
  model_path="'/abs/path/to/checkpoint'" \
  num_gpus=1 batch_size=4 server_url=0.0.0.0 server_port=8010
```

> Note the inner quotes around `model_path`: Hydra treats `=` as an override
> separator, and checkpoint directories commonly contain `=` (e.g.
> `ckpt-avg-2metrics=0.9219_step=750`). Without the quotes Hydra fails with
> `mismatched input '=' expecting <EOF>`.

### 2. Reach the server

The server binds `0.0.0.0` on a compute node. Either run the client from inside
a job on the same node, or tunnel:

```bash
ssh -N -L 8010:<compute-node>:8010 <user>@<cluster>
```

### 3. Send videos

Multi-view (server tiles automatically):

```bash
python scripts/send_videos_to_tiled_eval.py \
  --url http://localhost:8010 \
  --task "push the green button" \
  --view top=top.mp4 --view front=front.mp4 --view wrist=wrist.mp4
```

Already-tiled single video (no tiling applied):

```bash
python scripts/send_videos_to_tiled_eval.py \
  --url http://localhost:8010 \
  --task "push the green button" --tiled-video tiled.mp4
```

Output:

```text
POST http://localhost:8010/evaluate_videos  views=['top', 'front', 'wrist']  task='push the green button'
server tiling: primary=top secondary=['front', 'wrist']
frames=32
per-frame reward:
  0.008 0.169 0.305 ... 0.631 0.173
final reward: 0.1727   mean: 0.4884
final success prob: 0.0012
```

Add `--json-out rewards.json` to keep the full response, `--frame-steps` to score
progressive prefixes (`0:1`, `0:2`, ...) and aggregate them, and `--max-frames`,
`--max-width`, `--target-width`, `--target-height`, `--primary-view`,
`--secondary-views` to control decoding and layout.

### curl equivalent

```bash
curl -sS -X POST http://localhost:8010/evaluate_videos \
  -F "task=push the green button" \
  -F "max_frames=32" \
  -F "view_top=@top.mp4" \
  -F "view_front=@front.mp4" \
  -F "view_wrist=@wrist.mp4"
```

## Endpoint

`POST /evaluate_videos` — multipart form.

| Field | Meaning |
| --- | --- |
| `task` | **Required.** Language instruction shown to the model. |
| one file per view | Field name (or filename stem) is the view name: `view_top`, `top`, `top.mp4` all mean `top`. |
| `primary_view` | View placed in the large upper panel. Default: first of `top`, `front`, `agentview`, `left`. |
| `secondary_views` | Comma-separated lower-grid order. Default: all other views. |
| `target_width` / `target_height` | Canvas size. Omitted → inferred from the primary view's aspect ratio. |
| `primary_height_fraction` | Upper-panel share of canvas height. Default `2/3`. |
| `max_frames` | Frames sampled per view. Default `64`. |
| `max_width` | Optional downscale before tiling. |
| `use_frame_steps` | Score progressive prefixes and aggregate. Default `false`. |

Single upload with no `secondary_views` is treated as **already tiled** and is
passed through untouched (`tiling.mode == "pre_tiled"`).

Response:

```json
{
  "outputs_progress": {"progress_pred": [[...per-frame rewards...]]},
  "outputs_success": {"success_probs": [[...]]},
  "num_frames": 32,
  "tiling": {"mode": "tiled_by_server", "primary_view": "top",
             "secondary_views": ["front", "wrist"],
             "source_shapes": {"top": [32, 240, 426, 3], "...": "..."}}
}
```

Other endpoints are unchanged: `/evaluate_batch`, `/evaluate_batch_npy`,
`/health`, `/gpu_status`, `/model_info`.

## Layout expectations

Checkpoints are trained on a specific view layout, so match it when calling:

| Checkpoint family | Primary | Secondary (lower grid) | Canvas |
| --- | --- | --- | --- |
| SO-101 busybox / Armnet single-arm | `top` | `front`, `wrist` | 640×540 |
| Armnet bimanual | `top` | `left_wrist`, `right_wrist` | 640×540 |
| RoboCasa365 | `left` | `right`, `wrist` | 512×768 |
| MolmoAct2 YAM | `top` | `left`, `right` | 640×540 |

Mismatched views still produce numbers but the rewards are not meaningful.

## More views than panels

The lower grid is built automatically: `ceil(sqrt(n))` columns, panels sized to
fit. Unused cells are filled with black. Panels are letterbox-free resizes, so
keep `target_width`/`target_height` proportional to your source aspect (640×540
matches 426×240 sources almost exactly: 640×360 primary + 320×180 cells).

## Troubleshooting

**`mismatched input '=' expecting <EOF>`** — quote the model path:
`model_path="'/abs/path/with=equals'"`.

**Client hangs / 502 on loopback** — clusters often export `http_proxy` without
`127.0.0.1` in `no_proxy`, so loopback requests go to the proxy. The provided
client disables proxy env vars for loopback hosts; for `curl` use
`--noproxy '*'`, and set `NO_PROXY=localhost,127.0.0.1` in your shell.

**`module 'pydantic.v1' has no attribute 'BaseModel'`** — a corrupted
`pydantic/v1` subpackage (it should contain ~27 files). Restore it from the
pinned wheel or reinstall `pydantic==2.13.4`; every FastAPI app fails otherwise.

**`Input type (CUDABFloat16Type) and weight type (CPUBFloat16Type)`** — the
frozen vision tower of a deep-copied replica still pointed at the original CPU
module. `eval_server._rebind_frozen_visual` fixes this after the device move; if
you copy models elsewhere, rebind there too.

**Job dies in seconds on `g002`** — exclude it: `#SBATCH --exclude=g002`.

**Job pending forever** — `weirdlab` has low fairshare; `rselab` rejects jobs
over its budget. Check `squeue --start` before assuming it is stuck.

## Verified checkpoint

`jesbu1/robometer-4b-fft-busybox-tiled` (FFT from `robometer/Robometer-4B`,
2,900 episodes of `push_green_button`, 1,450 successes / 1,450 timeouts):

- training-time: Pearson 0.844, Kendall 1.000, ranking acc 1.000
- standalone eval: Pearson 0.974, Kendall 0.996, ranking acc 0.998
