import os
import subprocess as sp
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    create_trajectory_video,
    generate_unique_id,
    load_sentence_transformer_model,
)
from robometer.utils.tiling import tile_synchronized_views

TILED_VIEWS = ("top", "left", "right")


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


def _make_frame_loader(video_path: str, from_ts: float, to_ts: float, target_fps: int = 10):
    def _load():
        duration = to_ts - from_ts
        if duration <= 0:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)

        width, height = _probe_video_dims(video_path)

        cmd = [
            "ffmpeg",
            "-ss", str(from_ts),
            "-i", video_path,
            "-t", str(duration),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-r", str(target_fps),
            "-v", "error",
            "-",
        ]
        raw = sp.check_output(cmd)
        frame_bytes = width * height * 3
        n = len(raw) // frame_bytes
        if n == 0:
            return np.empty((0, height, width, 3), dtype=np.uint8)
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(n, height, width, 3)
        return frames

    return _load


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    shard_index = index // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_video_output_path(
    output_dir: str,
    dataset_label: str,
    trajectory_idx: int,
) -> tuple[str, str]:
    shard_dir = _stable_shard_for_index(trajectory_idx)
    traj_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"trajectory_{trajectory_idx:06d}")
    os.makedirs(traj_dir, exist_ok=True)
    filename = "trajectory.mp4"
    full_path = os.path.join(traj_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"trajectory_{trajectory_idx:06d}", filename)
    return full_path, rel_path


def convert_molmoact2_yam_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
) -> Dataset:
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    meta_dir = root / "meta"
    videos_dir = root / "videos"
    if not meta_dir.exists():
        raise ValueError(f"No meta/ directory found under {root} — expected LeRobot v3.0 format")

    print(f"Dataset root: {root}")

    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    entries: list[dict] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    VIEW_KEY = "observation.images.top"

    tasks_path = meta_dir / "tasks_annotated.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")
    tasks_df = pd.read_parquet(tasks_path)
    ep_to_task: dict[int, str] = {}
    for ep_idx, row in tasks_df.iterrows():
        task_text = str(row.get("task", "")).strip()
        if task_text:
            ep_to_task[int(ep_idx)] = task_text
    print(f"Loaded {len(ep_to_task)} task annotations")

    ep_parquet_dir = meta_dir / "episodes"
    ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode parquet files found under {ep_parquet_dir}")

    ep_cols_needed = ["episode_index"]
    for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]:
        ep_cols_needed.append(f"videos/{VIEW_KEY}/{suffix}")

    ep_rows = []
    for f in ep_files:
        table = pq.read_table(f, columns=ep_cols_needed)
        names = table.column_names
        for batch in table.to_batches():
            for i in range(batch.num_rows):
                ep_rows.append({name: batch.column(j)[i].as_py() for j, name in enumerate(names)})

    episodes_df = pd.DataFrame(ep_rows)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    total_episodes = len(episodes_df)
    print(f"Found {total_episodes} episodes across {len(ep_files)} parquet file(s)")

    for _, ep_row in tqdm(episodes_df.iterrows(), total=total_episodes, desc="Processing episodes"):
        if produced >= max_limit:
            break

        ep_idx = int(ep_row["episode_index"])
        task_text = ep_to_task.get(ep_idx)
        if not task_text:
            continue

        if task_text not in lang_cache:
            lang_cache[task_text] = lang_model.encode(task_text)
        lang_vec = lang_cache[task_text]

        v_chunk = int(ep_row[f"videos/{VIEW_KEY}/chunk_index"])
        v_file = int(ep_row[f"videos/{VIEW_KEY}/file_index"])
        from_ts = float(ep_row[f"videos/{VIEW_KEY}/from_timestamp"])
        to_ts = float(ep_row[f"videos/{VIEW_KEY}/to_timestamp"])

        video_path = (
            videos_dir / VIEW_KEY / f"chunk-{v_chunk:03d}" / f"file-{v_file:03d}.mp4"
        )
        if not video_path.exists():
            continue

        full_video_path, rel_video_path = _build_video_output_path(
            output_dir=output_dir,
            dataset_label=dataset_name,
            trajectory_idx=produced,
        )

        frame_loader = _make_frame_loader(str(video_path), from_ts, to_ts, target_fps=fps)

        traj_dict = {
            "id": generate_unique_id(),
            "frames": frame_loader,
            "task": task_text,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }

        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=full_video_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=True,
            fps=fps,
        )
        if entry:
            entry["frames"] = rel_video_path
            entries.append(entry)
            produced += 1

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

    The source videos are all-keyframe AV1, so decoding is cheap and seek-accurate.
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
        chunk = int(ep_row[f"videos/observation.images.{view}/chunk_index"])
        file_index = int(ep_row[f"videos/observation.images.{view}/file_index"])
        from_ts = float(ep_row[f"videos/observation.images.{view}/from_timestamp"])
        to_ts = float(ep_row[f"videos/observation.images.{view}/to_timestamp"])
        video_path = (
            root / "videos" / f"observation.images.{view}" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
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


def convert_molmoact2_yam_tiled_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 32,
    fps: int = 10,
    target_width: int = 640,
    target_height: int = 540,
    num_workers: int = 12,
) -> Dataset:
    """Convert MolmoAct2 YAM episodes into tiled (top/left/right) Robometer trajectories.

    The ``top`` view occupies the primary upper panel; ``left`` and ``right`` share
    the lower grid. Output videos are ``target_width x target_height`` at ``fps``.
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    meta_dir = root / "meta"
    tasks_path = meta_dir / "tasks_annotated.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    tasks_df = pd.read_parquet(tasks_path)
    ep_to_task: dict[int, str] = {
        int(ep_idx): str(row["task"]).strip()
        for ep_idx, row in tasks_df.iterrows()
        if str(row.get("task", "")).strip()
    }
    print(f"Loaded {len(ep_to_task)} task annotations")

    ep_parquet_dir = meta_dir / "episodes"
    ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode parquet files found under {ep_parquet_dir}")

    ep_cols_needed = ["episode_index"]
    for view in TILED_VIEWS:
        for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]:
            ep_cols_needed.append(f"videos/observation.images.{view}/{suffix}")

    ep_rows = []
    for f in ep_files:
        table = pq.read_table(f, columns=ep_cols_needed)
        names = table.column_names
        for batch in table.to_batches():
            for i in range(batch.num_rows):
                ep_rows.append({name: batch.column(j)[i].as_py() for j, name in enumerate(names)})

    episodes_df = pd.DataFrame(ep_rows).sort_values("episode_index").reset_index(drop=True)
    print(f"Found {len(episodes_df)} episodes across {len(ep_files)} parquet file(s)")

    # Probe per-view dimensions once from the first available video
    probe_dims: dict[str, tuple[int, int]] = {}
    for view in TILED_VIEWS:
        for video_file in sorted((root / "videos" / f"observation.images.{view}").rglob("*.mp4"))[:1]:
            probe_dims[view] = _probe_video_dims(str(video_file))
    missing_views = [view for view in TILED_VIEWS if view not in probe_dims]
    if missing_views:
        raise ValueError(f"Could not probe dimensions for views: {missing_views}")
    print(f"View dimensions: {probe_dims}")

    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    def process_episode(args: tuple[int, dict]) -> dict | None:
        produced_idx, ep_row = args
        ep_idx = int(ep_row["episode_index"])
        task_text = ep_to_task.get(ep_idx)
        if not task_text:
            return None
        frames_by_view = _load_tiled_episode(ep_row, root, fps, max_frames, probe_dims)
        if frames_by_view is None:
            return None
        tiled = tile_synchronized_views(
            frames_by_view,
            primary_view="top",
            secondary_views=["left", "right"],
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
    work_items = [
        (idx, row)
        for idx, (_, row) in enumerate(episodes_df.iterrows())
        if idx < max_limit
    ]
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
