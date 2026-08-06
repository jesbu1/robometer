import os
import subprocess as sp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)


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

    print(f"Total entries produced: {len(entries)}")
    return Dataset.from_list(entries)
