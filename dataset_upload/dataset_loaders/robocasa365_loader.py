"""RoboCasa365 tiled dataset converter.

Converts the LeRobot v3.0 RoboCasa365 pretraining corpora (human teleoperation,
atomic + composite subsets) into tiled Robometer trajectories. The ``left``
agentview occupies the primary upper panel; ``right`` and ``wrist`` share the
lower grid. Source frames are 256x256 h264 at 20 fps; output videos are
``target_width x target_height`` (default 512x768, preserving square panels)
with ``max_frames`` sampled at ``fps`` (default 32 @ 10 fps).
"""

import os
import subprocess as sp
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from dataset_upload.helpers import (
    create_trajectory_video,
    generate_unique_id,
    load_sentence_transformer_model,
)
from datasets import Dataset
from robometer.utils.tiling import tile_synchronized_views

TILED_VIEWS = ("left", "right", "wrist")
CAMERA_KEYS = {
    "left": "observation.images.robot0_agentview_left",
    "right": "observation.images.robot0_agentview_right",
    "wrist": "observation.images.robot0_eye_in_hand",
}
PRIMARY_VIEW = "left"
SECONDARY_VIEWS = ["right", "wrist"]


def _probe_video_dims(video_path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path,
    ]
    out = sp.check_output(cmd).decode().strip()
    parts = out.split(",")
    return int(parts[0]), int(parts[1])


def _read_segment_frames(
    video_path: str,
    from_ts: float,
    to_ts: float,
    fps: int,
    max_frames: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Decode one episode segment and keep at most ``max_frames`` uniformly spaced frames.

    Frames are streamed from ffmpeg and only the sampled targets are retained to
    keep memory bounded regardless of segment length.
    """
    duration = to_ts - from_ts
    if duration <= 0 or not os.path.exists(video_path):
        return np.empty((0, height, width, 3), dtype=np.uint8)

    expected_output = max(1, round(duration * fps))
    keep_count = min(max_frames, expected_output)
    if keep_count < 2:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    target_indices = set(np.linspace(0, expected_output - 1, keep_count, dtype=int).tolist())

    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-threads", "2",
        "-ss", f"{from_ts:.6f}",
        "-i", video_path,
        "-t", f"{duration:.6f}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-",
    ]
    kept: list[np.ndarray] = []
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)
    assert proc.stdout is not None
    index = 0
    try:
        while index <= max(target_indices, default=-1):
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if index in target_indices:
                kept.append(np.frombuffer(buf, dtype=np.uint8).copy().reshape(height, width, 3))
            index += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    if not kept:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    return np.stack(kept)


def _load_tiled_episode(
    ep_row: dict,
    root: Path,
    fps: int,
    max_frames: int,
    probe_dims: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray] | None:
    """Decode and return frames for each camera view of one episode, or None.

    The three camera decodes run in parallel threads within the worker.
    """
    results: dict[str, np.ndarray | None] = {}
    threads: list[threading.Thread] = []

    def decode_view(view: str) -> None:
        camera = CAMERA_KEYS[view]
        chunk = int(ep_row[f"videos/{camera}/chunk_index"])
        file_index = int(ep_row[f"videos/{camera}/file_index"])
        from_ts = float(ep_row[f"videos/{camera}/from_timestamp"])
        to_ts = float(ep_row[f"videos/{camera}/to_timestamp"])
        video_path = (
            root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        )
        if not video_path.exists():
            results[view] = None
            return
        width, height = probe_dims[view]
        frames = _read_segment_frames(str(video_path), from_ts, to_ts, fps, max_frames, width, height)
        results[view] = frames if len(frames) else None

    for view in TILED_VIEWS:
        t = threading.Thread(target=decode_view, args=(view,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if any(results.get(view) is None for view in TILED_VIEWS):
        return None
    return {view: results[view] for view in TILED_VIEWS}


def _load_source_episodes(source: Path) -> pd.DataFrame:
    """Read episode metadata (with tasks) from one LeRobot v3.0 source tree."""
    meta_dir = source / "meta"
    ep_parquet_dir = meta_dir / "episodes"
    ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode parquet files found under {ep_parquet_dir}")

    ep_cols_needed = ["episode_index", "tasks"]
    for view in TILED_VIEWS:
        camera = CAMERA_KEYS[view]
        for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]:
            ep_cols_needed.append(f"videos/{camera}/{suffix}")

    ep_rows = []
    for f in ep_files:
        table = pq.read_table(f, columns=ep_cols_needed)
        names = table.column_names
        for batch in table.to_batches():
            for i in range(batch.num_rows):
                ep_rows.append({name: batch.column(j)[i].as_py() for j, name in enumerate(names)})

    episodes_df = pd.DataFrame(ep_rows).sort_values("episode_index").reset_index(drop=True)
    print(f"Found {len(episodes_df)} episodes across {len(ep_files)} parquet file(s) in {source}")
    return episodes_df


def convert_robocasa365_tiled_to_hf(
    dataset_path: str | list[str],
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 32,
    fps: int = 10,
    target_width: int = 512,
    target_height: int = 768,
    num_workers: int = 12,
) -> Dataset:
    """Convert RoboCasa365 pretraining episodes into tiled Robometer trajectories.

    ``dataset_path`` may be a single LeRobot v3.0 source directory, a list of
    them, or a comma-separated string; sources are processed in order and the
    trajectory index continues across sources. The ``left`` agentview occupies
    the primary upper panel; ``right`` and ``wrist`` share the lower grid.
    """
    if isinstance(dataset_path, str):
        parts = [p for p in dataset_path.split(",") if p.strip()]
    else:
        parts = [str(p) for p in dataset_path]
    sources = [Path(os.path.expanduser(p)) for p in parts]
    for source in sources:
        if not (source / "meta" / "episodes").exists():
            raise FileNotFoundError(
                f"No meta/episodes under {source} — expected LeRobot v3.0 format"
            )
    print(f"Source datasets: {[str(s) for s in sources]}")

    # Probe per-view dimensions once from the first available video
    probe_dims: dict[str, tuple[int, int]] = {}
    for view in TILED_VIEWS:
        for video_file in sorted((sources[0] / "videos" / CAMERA_KEYS[view]).rglob("*.mp4"))[:1]:
            probe_dims[view] = _probe_video_dims(str(video_file))
    missing_views = [view for view in TILED_VIEWS if view not in probe_dims]
    if missing_views:
        raise ValueError(f"Could not probe dimensions for views: {missing_views}")
    print(f"View dimensions: {probe_dims}")

    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    def process_episode(args: tuple[int, Path, dict]) -> dict | None:
        produced_idx, root, ep_row = args
        task_list = ep_row.get("tasks") or []
        task_text = next((str(t).strip() for t in task_list if str(t).strip()), None)
        if not task_text:
            return None
        frames_by_view = _load_tiled_episode(ep_row, root, fps, max_frames, probe_dims)
        if frames_by_view is None:
            return None
        tiled = tile_synchronized_views(
            frames_by_view,
            primary_view=PRIMARY_VIEW,
            secondary_views=SECONDARY_VIEWS,
            target_width=target_width,
            target_height=target_height,
        )
        traj_dir = os.path.join(
            output_dir, dataset_name.lower(), f"shard_{produced_idx // 1000:04d}", f"trajectory_{produced_idx:06d}"
        )
        os.makedirs(traj_dir, exist_ok=True)
        create_trajectory_video(
            tiled,
            traj_dir,
            max_frames=max_frames,
            fps=fps,
            shortest_edge_size=min(target_width, target_height),
            center_crop=False,
        )
        if task_text not in lang_cache:
            lang_cache[task_text] = lang_model.encode(task_text)
        rel_path = os.path.join(
            dataset_name.lower(),
            f"shard_{produced_idx // 1000:04d}",
            f"trajectory_{produced_idx:06d}",
            "trajectory.mp4",
        )
        return {
            "id": generate_unique_id(),
            "task": task_text,
            "lang_vector": lang_cache[task_text],
            "data_source": dataset_name,
            "frames": rel_path,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
            "partial_success": None,
        }

    entries: list[dict] = []
    work_items: list[tuple[int, Path, dict]] = []
    produced = 0
    for source in sources:
        episodes_df = _load_source_episodes(source)
        for _, row in episodes_df.iterrows():
            if produced >= max_limit:
                break
            work_items.append((produced, source, row))
            produced += 1
        if produced >= max_limit:
            break

    skipped = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for result in tqdm(pool.map(process_episode, work_items), total=len(work_items), desc=f"Tiling {dataset_name}"):
            if result is None:
                skipped += 1
            else:
                entries.append(result)

    print(f"Total entries produced: {len(entries)} (skipped {skipped} episodes)")
    if not entries:
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
            "preference_group_id": [],
            "preference_rank": [],
        })
    return Dataset.from_list(entries)
