"""Convert native Armnet LeRobot episodes into tiled Robometer trajectories."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import (
    create_trajectory_video,
    generate_unique_id,
    load_sentence_transformer_model,
    tile_synchronized_views,
)


def _read_video(path: Path, start: float, end: float, fps: int, max_frames: int) -> np.ndarray:
    """Decode one episode segment as RGB frames."""
    if end <= start or not path.exists():
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        seek_timestamp = max(0, int(start / float(stream.time_base)))
        container.seek(seek_timestamp, stream=stream, backward=True, any_frame=False)
        target_count = max(1, min(max_frames, int(np.ceil((end - start) * fps))))
        targets = np.linspace(start, end, target_count, endpoint=False)
        target_index = 0
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None:
                continue
            if timestamp < start:
                continue
            if timestamp >= end:
                break
            if target_index < target_count and timestamp >= targets[target_index]:
                frames.append(frame.to_ndarray(format="rgb24"))
                target_index += 1
                if target_index == target_count:
                    break

    if not frames:
        return np.empty((0, stream.height, stream.width, 3), dtype=np.uint8)
    return np.stack(frames)


def _quality_label(row: pd.Series) -> str:
    value = str(row.get("success_class", "")).strip().lower()
    if value in {"successful", "failure", "suboptimal"}:
        return value
    raise ValueError(f"Unknown Armnet success_class={value!r} for episode {row['episode_index']}")


def _video_path(root: Path, camera: str, row: pd.Series) -> Path:
    chunk = int(row[f"videos/{camera}/chunk_index"])
    file_index = int(row[f"videos/{camera}/file_index"])
    return root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def _episode_rows(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata found under {root / 'meta' / 'episodes'}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True).sort_values("episode_index")


def _empty_dataset() -> Dataset:
    return Dataset.from_dict({
        "id": [], "task": [], "lang_vector": [], "data_source": [], "frames": [],
        "is_robot": [], "quality_label": [], "partial_success": [],
    })


def convert_armnet_lerobot_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 20,
    target_width: int | None = None,
    target_height: int | None = None,
) -> Dataset:
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Armnet dataset path not found: {root}")

    target_width = target_width or 640
    target_height = target_height or 540

    if "bimanual" in dataset_name.lower():
        cameras = ("observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist")
    else:
        cameras = ("observation.images.top", "observation.images.front", "observation.images.wrist")

    episodes = _episode_rows(root)
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_map = {int(row["task_index"]): str(index) for index, row in tasks.iterrows()}
    language_model = load_sentence_transformer_model()
    language_cache: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    limit = None if max_trajectories in (None, -1) else max_trajectories
    decoder_pool = ThreadPoolExecutor(max_workers=3)

    print(f"Armnet episodes: {len(episodes)} | cameras: {cameras} | target: {target_width}x{target_height}")
    print("Native labels:")
    print(episodes["success_class"].value_counts(dropna=False).to_string())

    for _, row in tqdm(episodes.iterrows(), total=len(episodes), desc=f"Tiling {dataset_name}"):
        if limit is not None and len(entries) >= limit:
            break
        task_value = row.get("tasks")
        if isinstance(task_value, (list, tuple, np.ndarray)):
            task = str(task_value[0]) if len(task_value) else ""
        else:
            task = str(task_value or "")
        if not task:
            task = task_map.get(int(row.get("task_index", -1)), "")
        if not task:
            raise ValueError(f"Missing task for episode {row['episode_index']}")

        label = _quality_label(row)
        if task not in language_cache:
            language_cache[task] = language_model.encode(task)

        decoded = list(decoder_pool.map(
            lambda camera: _read_video(
                _video_path(root, camera, row),
                float(row[f"videos/{camera}/from_timestamp"]),
                float(row[f"videos/{camera}/to_timestamp"]),
                fps,
                max_frames,
            ),
            cameras,
        ))
        frame_count = min(len(frames) for frames in decoded)
        if frame_count == 0:
            raise ValueError(f"No synchronized frames for episode {row['episode_index']}")
        tiled = tile_synchronized_views(
            dict(zip(cameras, (frames[:frame_count] for frames in decoded), strict=True)),
            primary_view=cameras[0],
            secondary_views=cameras[1:],
            target_width=target_width,
            target_height=target_height,
        )

        trajectory_index = len(entries)
        relative_path = os.path.join(
            dataset_name.lower(), f"shard_{trajectory_index // 1000:04d}",
            f"trajectory_{trajectory_index:06d}", "trajectory.mp4",
        )
        full_path = Path(output_dir) / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        create_trajectory_video(
            tiled,
            str(full_path.parent),
            max_frames=max_frames,
            fps=fps,
            shortest_edge_size=min(target_width, target_height),
            center_crop=False,
        )
        entries.append({
            "id": generate_unique_id(),
            "task": task,
            "lang_vector": language_cache[task],
            "data_source": "armnetbench_tiled",
            "frames": relative_path,
            "is_robot": True,
            "quality_label": label,
            "partial_success": None,
        })

    decoder_pool.shutdown(wait=True)
    return Dataset.from_list(entries) if entries else _empty_dataset()
