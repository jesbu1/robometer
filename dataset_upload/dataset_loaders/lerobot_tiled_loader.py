"""Generic LeRobot v3.0 -> tiled Robometer converter.

Works for any LeRobot v3.0 tree whose episodes contain synchronized camera
streams: camera views are discovered from ``meta/info.json`` and tiled with
``robometer.utils.tiling.tile_synchronized_views`` into one frame per timestep.

Episode quality is derived from a per-frame success column in the data parquet
(e.g. ``success``), falling back to a ``labels.jsonl`` sidecar keyed by
``episode_index``. Missing labels raise instead of silently defaulting to
successful.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import (
    create_trajectory_video,
    generate_unique_id,
    load_sentence_transformer_model,
)
from robometer.utils.tiling import tile_synchronized_views

# Preference order for choosing the primary (large) view and the lower grid.
PRIMARY_PREFERENCE = ("top", "front", "agentview", "left", "right")
SECONDARY_PREFERENCE = ("front", "left", "right", "wrist", "top", "agentview")


def discover_cameras(info: dict[str, Any]) -> list[str]:
    """Return the ``observation.images.*`` camera keys declared in meta/info.json."""
    features = info.get("features") or {}
    cameras = [key for key in features if str(key).startswith("observation.images.")]
    if not cameras:
        raise ValueError("No observation.images.* features found in meta/info.json")
    return sorted(cameras)


def short_view_name(camera_key: str) -> str:
    return camera_key.split("observation.images.", 1)[-1]


def choose_views(camera_keys: list[str]) -> tuple[str, list[str]]:
    """Pick a primary view plus ordered secondary views from camera keys."""
    views = [short_view_name(key) for key in camera_keys]
    primary = next((name for name in PRIMARY_PREFERENCE if name in views), views[0])
    ordered = [name for name in SECONDARY_PREFERENCE if name in views and name != primary]
    ordered += [name for name in views if name != primary and name not in ordered]
    return primary, ordered


def infer_target_size(
    primary_width: int,
    primary_height: int,
    secondary_count: int,
    primary_height_fraction: float = 2 / 3,
) -> tuple[int, int]:
    """Choose a canvas that keeps every panel close to its native aspect ratio.

    The lower grid uses ``ceil(sqrt(n))`` columns, so the canvas width is scaled
    up until each lower cell can hold the primary view's aspect ratio.
    """
    from math import ceil, sqrt

    columns = max(1, ceil(sqrt(max(1, secondary_count))))
    rows = max(1, ceil(max(1, secondary_count) / columns))
    aspect = primary_width / primary_height
    # lower cell aspect: (width/columns) / ((1-f)*height) == aspect
    width_for_cells = aspect * (1 - primary_height_fraction) * rows * columns
    width = max(float(primary_width), width_for_cells)
    height = width / aspect / primary_height_fraction
    return int(round(width)), int(round(height))


def _episode_rows(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta' / 'episodes'}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def _episode_success(root: Path, success_column: str) -> pd.Series:
    """Per-episode success label from data parquet, else labels.jsonl."""
    data_files = sorted((root / "data").rglob("*.parquet"))
    if data_files:
        frames = [
            pd.read_parquet(path, columns=["episode_index", success_column])
            for path in data_files
        ]
        data = pd.concat(frames, ignore_index=True)
        labels = data.groupby("episode_index")[success_column].max()
        return labels.astype(bool)

    labels_path = root / "labels.jsonl"
    if labels_path.exists():
        rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        if {"episode_index", "success"} <= set(frame.columns):
            return frame.set_index("episode_index")["success"].astype(bool)

    raise ValueError(
        f"No per-episode labels found: neither data/*.parquet column {success_column!r} "
        f"nor labels.jsonl in {root}"
    )


def _probe_dims(video_path: Path) -> tuple[int, int]:
    import subprocess as sp

    out = sp.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path),
        ]
    ).decode().strip()
    width, height = out.split(",")
    return int(width), int(height)


def _read_segment(
    video_path: Path,
    from_ts: float,
    to_ts: float,
    fps: int,
    max_frames: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Decode one camera segment, keeping at most ``max_frames`` sampled frames."""
    import subprocess as sp

    duration = to_ts - from_ts
    if duration <= 0 or not video_path.exists():
        return np.empty((0, height, width, 3), dtype=np.uint8)

    expected = max(1, round(duration * fps))
    keep = min(max_frames, expected)
    if keep < 2:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    targets = set(np.linspace(0, expected - 1, keep, dtype=int).tolist())

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2",
        "-ss", f"{from_ts:.6f}", "-i", str(video_path), "-t", f"{duration:.6f}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-r", str(fps), "-",
    ]
    frame_bytes = width * height * 3
    kept: list[np.ndarray] = []
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)
    assert proc.stdout is not None
    index = 0
    try:
        while index <= max(targets, default=-1):
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if index in targets:
                kept.append(np.frombuffer(buf, dtype=np.uint8).copy().reshape(height, width, 3))
            index += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if not kept:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    return np.stack(kept)


def convert_lerobot_tiled_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    target_width: int | None = None,
    target_height: int | None = None,
    num_workers: int = 8,
    success_column: str = "success",
    primary_view: str | None = None,
    secondary_views: list[str] | None = None,
    quality_column_name: str = "quality_label",
) -> Dataset:
    """Convert a LeRobot v3.0 tree into tiled Robometer trajectories.

    Views are auto-discovered from ``meta/info.json`` unless overridden. Output
    canvas size is inferred from the primary view's aspect ratio when omitted.
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"LeRobot dataset path not found: {root}")

    info = json.loads((root / "meta" / "info.json").read_text())
    cameras = discover_cameras(info)
    view_to_camera = {short_view_name(key): key for key in cameras}
    auto_primary, auto_secondary = choose_views(cameras)
    primary_view = primary_view or auto_primary
    secondary = list(secondary_views or auto_secondary)
    if primary_view not in view_to_camera:
        raise KeyError(f"Primary view {primary_view!r} not among discovered views {list(view_to_camera)}")
    secondary = [name for name in secondary if name != primary_view]
    if not secondary:
        raise ValueError("At least one secondary view is required")

    episodes = _episode_rows(root)
    success = _episode_success(root, success_column)

    probe_path = sorted((root / "videos" / view_to_camera[primary_view]).rglob("*.mp4"))
    if not probe_path:
        raise FileNotFoundError(f"No videos found for primary view {primary_view!r}")
    primary_w, primary_h = _probe_dims(probe_path[0])
    if target_width is None or target_height is None:
        inferred_w, inferred_h = infer_target_size(primary_w, primary_h, len(secondary))
        target_width = int(target_width or inferred_w)
        target_height = int(target_height or inferred_h)

    dims: dict[str, tuple[int, int]] = {primary_view: (primary_w, primary_h)}
    for name in secondary:
        vids = sorted((root / "videos" / view_to_camera[name]).rglob("*.mp4"))
        if not vids:
            raise FileNotFoundError(f"No videos found for view {name!r}")
        dims[name] = _probe_dims(vids[0])

    print(
        f"LeRobot tiled conversion: {len(episodes)} episodes | primary={primary_view} "
        f"secondary={secondary} | source {primary_w}x{primary_h} | output {target_width}x{target_height}"
    )
    print(f"Label distribution: {success.value_counts().to_dict()}")

    language_model = load_sentence_transformer_model()
    language_cache: dict[str, Any] = {}
    limit = None if max_trajectories in (None, -1) else int(max_trajectories)

    work: list[tuple[int, pd.Series, str, str]] = []
    for _, row in episodes.iterrows():
        if limit is not None and len(work) >= limit:
            break
        episode_index = int(row["episode_index"])
        task_value = row.get("tasks")
        if isinstance(task_value, (list, tuple, np.ndarray)):
            task = str(task_value[0]).strip() if len(task_value) else ""
        else:
            task = str(task_value or "").strip()
        if not task:
            raise ValueError(f"Missing task text for episode {episode_index}")
        label = "successful" if bool(success.get(episode_index, False)) else "failure"
        work.append((len(work), row, task, label))

    def process(item: tuple[int, pd.Series, str, str]) -> dict[str, Any] | None:
        produced_idx, row, task, label = item
        decoded: dict[str, np.ndarray] = {}
        for name in [primary_view, *secondary]:
            camera = view_to_camera[name]
            chunk = int(row[f"videos/{camera}/chunk_index"])
            file_index = int(row[f"videos/{camera}/file_index"])
            width, height = dims[name]
            decoded[name] = _read_segment(
                root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4",
                float(row[f"videos/{camera}/from_timestamp"]),
                float(row[f"videos/{camera}/to_timestamp"]),
                fps,
                max_frames,
                width,
                height,
            )
            if len(decoded[name]) == 0:
                return None
        frame_count = min(len(value) for value in decoded.values())
        tiled = tile_synchronized_views(
            {name: value[:frame_count] for name, value in decoded.items()},
            primary_view=primary_view,
            secondary_views=secondary,
            target_width=target_width,
            target_height=target_height,
        )
        rel_path = os.path.join(
            dataset_name.lower(),
            f"shard_{produced_idx // 1000:04d}",
            f"trajectory_{produced_idx:06d}",
            "trajectory.mp4",
        )
        full_path = Path(output_dir) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        create_trajectory_video(
            tiled,
            str(full_path.parent),
            max_frames=max_frames,
            fps=fps,
            shortest_edge_size=min(target_width, target_height),
            center_crop=False,
        )
        if task not in language_cache:
            language_cache[task] = language_model.encode(task)
        return {
            "id": generate_unique_id(),
            "task": task,
            "lang_vector": language_cache[task],
            "data_source": dataset_name,
            "frames": rel_path,
            "is_robot": True,
            quality_column_name: label,
            "partial_success": None,
        }

    entries: list[dict[str, Any]] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for result in tqdm(pool.map(process, work), total=len(work), desc=f"Tiling {dataset_name}"):
            if result is None:
                skipped += 1
            else:
                entries.append(result)

    counts = pd.Series([entry[quality_column_name] for entry in entries]).value_counts().to_dict()
    print(f"Total entries: {len(entries)} (skipped {skipped}) | labels: {counts}")
    if not entries:
        return Dataset.from_dict({
            "id": [], "task": [], "lang_vector": [], "data_source": [], "frames": [],
            "is_robot": [], quality_column_name: [], "partial_success": [],
        })
    return Dataset.from_list(entries)
