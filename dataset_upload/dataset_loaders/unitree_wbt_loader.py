import json
import os
import random
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)

STEREO_CAMERAS = (
    "observation.images.head_stereo_left",
    "observation.images.head_stereo_right",
)


def _resolve_dataset_root(dataset_path: str, dataset_name: str) -> Path:
    """Resolve dataset root from either <dataset_path>/<dataset_name> or <dataset_path>."""
    base = Path(os.path.expanduser(dataset_path))
    candidates = [base / dataset_name, base]
    for candidate in candidates:
        if (candidate / "meta" / "info.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find dataset root in {base / dataset_name} or {base}. "
        "Expected meta/info.json to exist."
    )


def _load_episodes_metadata(root: Path) -> pd.DataFrame:
    """Load episode metadata from meta/episodes/ parquet files."""
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"No episodes metadata found at {episodes_dir}")

    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {episodes_dir}")

    dfs = [pd.read_parquet(p) for p in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def _read_task_text(root: Path, episodes_df: pd.DataFrame) -> str:
    """Get task text from tasks.parquet or episode metadata."""
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        if len(tasks_df) > 0:
            for col in tasks_df.columns:
                val = tasks_df.iloc[0][col]
                if isinstance(val, str) and val.strip():
                    return val.strip()

    if "tasks" in episodes_df.columns:
        task_val = episodes_df.iloc[0]["tasks"]
        if isinstance(task_val, np.ndarray) and len(task_val) > 0:
            return str(task_val[0]).strip()
        if isinstance(task_val, str) and task_val.strip():
            return task_val.strip()

    return ""


def _extract_frames_from_video(
    video_path: str,
    from_timestamp: float,
    to_timestamp: float,
    source_fps: float,
) -> list[np.ndarray]:
    """Read frames from a video file between two timestamps using pyav."""
    try:
        container = av.open(video_path)
    except Exception:
        return []

    frames: list[np.ndarray] = []
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base

        if from_timestamp > 0:
            seek_pts = int(from_timestamp / time_base)
            container.seek(seek_pts, stream=stream, any_frame=False)

        for frame in container.decode(video=0):
            ts = float(frame.pts * time_base)
            if ts < from_timestamp:
                continue
            if ts >= to_timestamp:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()

    return frames


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    shard_index = int(index) // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_video_paths(output_dir: str, dataset_label: str, episode_idx: int, camera_key: str) -> tuple[str, str]:
    shard_dir = _stable_shard_for_index(episode_idx)
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    camera_name = camera_key.split(".")[-1]
    filename = f"clip@{camera_name}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _choose_camera(available: list[str], seed_text: str) -> str:
    if len(available) == 1:
        return available[0]
    rng = random.Random(seed_text)
    return rng.choice(sorted(available))


def convert_unitree_wbt_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert a UnitreeWBT LeRobot dataset to HF format.

    Reads episode metadata and video files directly (no LeRobot API dependency)
    and randomly picks one of head_stereo_left / head_stereo_right per episode.
    """
    del num_workers

    dataset_root = _resolve_dataset_root(dataset_path, dataset_name)

    with open(dataset_root / "meta" / "info.json") as f:
        info = json.load(f)
    source_fps = float(info.get("fps", 30))
    video_path_template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    episodes_df = _load_episodes_metadata(dataset_root)

    task_text = _read_task_text(dataset_root, episodes_df)
    if not task_text:
        task_text = dataset_name.replace("_", " ")
    print(f"Task: {task_text}")

    lang_model = load_sentence_transformer_model()
    lang_vec = lang_model.encode(task_text)

    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)
    entries: list[dict[str, Any]] = []
    produced = 0

    for _, row in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc=f"Episodes ({dataset_name})"):
        if produced >= max_limit:
            break

        episode_idx = int(row["episode_index"])

        try:
            available_cameras: list[str] = []
            for cam in STEREO_CAMERAS:
                chunk_col = f"videos/{cam}/chunk_index"
                file_col = f"videos/{cam}/file_index"
                from_col = f"videos/{cam}/from_timestamp"
                if chunk_col not in row or file_col not in row or from_col not in row:
                    continue

                chunk_idx = int(row[chunk_col])
                file_idx = int(row[file_col])
                video_file = dataset_root / video_path_template.format(
                    video_key=cam, chunk_index=chunk_idx, file_index=file_idx
                )
                if video_file.exists():
                    available_cameras.append(cam)

            if not available_cameras:
                continue

            camera_key = _choose_camera(available_cameras, seed_text=f"{dataset_name}:{episode_idx}")

            chunk_idx = int(row[f"videos/{camera_key}/chunk_index"])
            file_idx = int(row[f"videos/{camera_key}/file_index"])
            from_ts = float(row[f"videos/{camera_key}/from_timestamp"])
            to_ts = float(row[f"videos/{camera_key}/to_timestamp"])

            video_file = str(
                dataset_root
                / video_path_template.format(video_key=camera_key, chunk_index=chunk_idx, file_index=file_idx)
            )

            frames = _extract_frames_from_video(video_file, from_ts, to_ts, source_fps)
            if not frames:
                print(f"  No frames decoded for episode {episode_idx}, skipping")
                continue

            full_path, rel_path = _build_video_paths(output_dir, dataset_name, episode_idx, camera_key)
            traj_dict = {
                "id": generate_unique_id(),
                "frames": frames,
                "task": task_text,
                "is_robot": True,
                "quality_label": "successful",
                "preference_group_id": None,
                "preference_rank": None,
            }

            entry = create_hf_trajectory(
                traj_dict=traj_dict,
                video_path=full_path,
                lang_vector=lang_vec,
                max_frames=max_frames,
                dataset_name=dataset_name,
                use_video=True,
                fps=fps,
            )
            if entry:
                entry["frames"] = rel_path
                entries.append(entry)
                produced += 1
        except Exception as e:
            print(f"Skipping episode {episode_idx}: {e}")
            continue

    print(f"Produced {produced} trajectories from {len(episodes_df)} episodes")

    if not entries:
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
        })

    return Dataset.from_list(entries)
