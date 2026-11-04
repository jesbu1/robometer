#!/usr/bin/env python3
"""Send multi-view videos to a tiled RBM eval server and print per-frame rewards.

No robometer dependency: the server decodes the videos, tiles the views into the
single-image layout the checkpoint was trained on, and returns progress/reward.

Examples:
  python scripts/send_videos_to_tiled_eval.py \
    --url http://localhost:8010 \
    --task "push the green button" \
    --view top=top.mp4 --view front=front.mp4 --view wrist=wrist.mp4

  # already-tiled single video, no tiling needed
  python scripts/send_videos_to_tiled_eval.py --url http://localhost:8010 \
    --task "push the green button" --tiled-video tiled.mp4

  # print final reward and save the full curve
  python scripts/send_videos_to_tiled_eval.py ... --json-out rewards.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests
from urllib.parse import urlsplit


def parse_view(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--view must be name=path, got {value!r}")
    name, path = value.split("=", 1)
    view = Path(path).expanduser()
    if not view.exists():
        raise argparse.ArgumentTypeError(f"video not found: {view}")
    return name.strip(), view


def main() -> int:
    parser = argparse.ArgumentParser(description="Query a tiled RBM eval server with raw view videos")
    parser.add_argument("--url", default="http://localhost:8010", help="eval server base URL")
    parser.add_argument("--task", required=True, help="language instruction for the trajectory")
    parser.add_argument(
        "--view",
        action="append",
        type=parse_view,
        default=[],
        metavar="NAME=MP4",
        help="one camera view video; repeatable (e.g. --view top=top.mp4)",
    )
    parser.add_argument("--tiled-video", type=Path, help="already-tiled single video (skip server tiling)")
    parser.add_argument("--primary-view", help="view used for the large upper panel")
    parser.add_argument("--secondary-views", help="comma-separated lower-grid order, e.g. front,wrist")
    parser.add_argument("--target-width", type=int, help="tiled canvas width")
    parser.add_argument("--target-height", type=int, help="tiled canvas height")
    parser.add_argument("--max-frames", type=int, default=64, help="cap on sampled frames per view")
    parser.add_argument("--max-width", type=int, help="optional downscale before tiling")
    parser.add_argument("--frame-steps", action="store_true", help="expand into frame-step sub-samples")
    parser.add_argument("--timeout", type=float, default=600.0, help="request timeout in seconds")
    parser.add_argument("--json-out", type=Path, help="write the full response JSON here")
    args = parser.parse_args()

    if not args.view and not args.tiled_video:
        parser.error("provide at least one --view NAME=MP4 or --tiled-video PATH")

    endpoint = args.url.rstrip("/") + "/evaluate_videos"
    data: dict[str, Any] = {"task": args.task, "max_frames": str(args.max_frames)}
    for key, value in (
        ("primary_view", args.primary_view),
        ("secondary_views", args.secondary_views),
        ("target_width", args.target_width),
        ("target_height", args.target_height),
        ("max_width", args.max_width),
    ):
        if value not in (None, ""):
            data[key] = str(value)
    if args.frame_steps:
        data["use_frame_steps"] = "true"

    files = []
    handles = []
    try:
        if args.tiled_video:
            handle = args.tiled_video.open("rb")
            handles.append(handle)
            files.append(("view_tiled", (args.tiled_video.name, handle, "video/mp4")))
        for name, path in args.view:
            handle = path.open("rb")
            handles.append(handle)
            files.append((f"view_{name}", (path.name, handle, "video/mp4")))

        session = requests.Session()
        host = (urlsplit(endpoint).hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            # HPC clusters often export http_proxy without 127.0.0.1 in no_proxy,
            # which silently breaks loopback requests.
            session.trust_env = False

        print(f"POST {endpoint}  views={[n for n, _ in args.view] or ['tiled']}  task={args.task!r}")
        response = session.post(endpoint, data=data, files=files, timeout=args.timeout)
    finally:
        for handle in handles:
            handle.close()

    if response.status_code != 200:
        print(f"HTTP {response.status_code}: {response.text[:2000]}", file=sys.stderr)
        return 1

    payload = response.json()
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")

    tiling = payload.get("tiling") or {}
    if tiling:
        print(f"server tiling: primary={tiling.get('primary_view')} secondary={tiling.get('secondary_views')}")
    print(f"frames={payload.get('num_frames')}")

    progress = payload.get("outputs_progress", {}).get("progress_pred") or payload.get("progress_pred") or []
    while progress and isinstance(progress[0], list):
        progress = progress[0]
    if progress:
        print("per-frame reward:")
        print("  " + " ".join(f"{float(value):.3f}" for value in progress))
        print(f"final reward: {float(progress[-1]):.4f}   mean: {sum(map(float, progress)) / len(progress):.4f}")
    else:
        print("no progress predictions in response")

    success = payload.get("outputs_success", {}).get("success_probs")
    while success and isinstance(success[0], list):
        success = success[0]
    if success:
        print(f"final success prob: {float(success[-1]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
