"""Utilities for tiling synchronized camera views into a single frame.

This is the canonical implementation used by dataset converters and by the
dynamic-tiling inference path (eval server / local inference scripts).
"""

import cv2
import numpy as np


def infer_tiled_dimensions(
    frames_by_view: dict[str, np.ndarray],
    primary_view: str,
    primary_height_fraction: float = 2 / 3,
) -> tuple[int, int]:
    """Infer an output canvas size for the given views.

    Uses the primary view's width and keeps its aspect behavior by scaling the
    canvas height so the primary panel (``primary_height_fraction`` of the canvas)
    preserves the primary view's aspect ratio. For a 640x360 primary view this
    yields the 640x540 canvas used by the tiled training datasets.
    """
    if primary_view not in frames_by_view:
        raise KeyError(f"Primary view {primary_view!r} not found in frames_by_view")
    primary = frames_by_view[primary_view]
    height, width = primary.shape[1], primary.shape[2]
    target_width = int(width)
    target_height = round(height / primary_height_fraction)
    return target_width, target_height


def apply_dynamic_tiling_to_trajectory(trajectory: dict) -> dict:
    """Tile a raw trajectory dict's ``views`` into ``frames`` in place.

    A trajectory payload may carry raw multi-view frames instead of pre-tiled
    ``frames`` so a tiled checkpoint can serve both input styles:

        trajectory = {
            "views": {"top": (T,H,W,3), "left": ..., "right": ...},
            "primary_view": "top",            # optional, defaults to first key
            "secondary_views": ["l", "r"],    # optional, defaults to remaining keys
            "target_width": 640,              # optional, inferred when omitted
            "target_height": 540,             # optional, inferred when omitted
            ...
        }

    The ``views``/layout keys are removed and ``frames``/``frames_shape`` are set
    to the tiled result. Returns the trajectory dict for chaining.
    """
    views = trajectory.pop("views", None)
    if not isinstance(views, dict) or not views:
        return trajectory
    from collections.abc import Mapping

    arrays = {}
    for name, value in views.items():
        if isinstance(value, Mapping) and value.get("__numpy_file__"):
            raise ValueError(
                f"View {name!r} still references numpy file {value['__numpy_file__']!r}; "
                "resolve __numpy_file__ refs before dynamic tiling"
            )
        arrays[name] = value
    primary_view = trajectory.pop("primary_view", None) or next(iter(arrays))
    secondary_views = trajectory.pop("secondary_views", None)
    fraction = float(trajectory.pop("primary_height_fraction", 2 / 3) or 2 / 3)
    target_width = trajectory.pop("target_width", None)
    target_height = trajectory.pop("target_height", None)
    if not target_width or not target_height:
        inferred_width, inferred_height = infer_tiled_dimensions(arrays, primary_view, fraction)
        target_width = int(target_width or inferred_width)
        target_height = int(target_height or inferred_height)
    tiled = tile_synchronized_views(
        arrays,
        primary_view=primary_view,
        target_width=int(target_width),
        target_height=int(target_height),
        secondary_views=secondary_views,
        primary_height_fraction=fraction,
    )
    trajectory["frames"] = tiled
    trajectory["frames_shape"] = tuple(tiled.shape)
    return trajectory


def tile_synchronized_views(
    frames_by_view: dict[str, np.ndarray],
    primary_view: str,
    target_width: int,
    target_height: int,
    secondary_views: list[str] | None = None,
    primary_height_fraction: float = 2 / 3,
    fill_value: int = 0,
) -> np.ndarray:
    """Tile synchronized RGB views with one large primary view and a lower grid.

    Args:
        frames_by_view: Mapping of view names to ``(T, H, W, 3)`` RGB arrays.
        primary_view: View that occupies the full-width upper panel.
        target_width: Output canvas width in pixels.
        target_height: Output canvas height in pixels.
        secondary_views: Ordered lower-panel views. Defaults to all other mapping keys.
        primary_height_fraction: Fraction of the canvas height used by the primary view.
        fill_value: Pixel value used for unused lower-grid cells.

    Returns:
        A ``(T, target_height, target_width, 3)`` uint8 RGB array.
    """
    if primary_view not in frames_by_view:
        raise KeyError(f"Primary view {primary_view!r} not found in frames_by_view")
    secondary = list(secondary_views or [key for key in frames_by_view if key != primary_view])
    if not secondary:
        raise ValueError("At least one secondary view is required")
    if any(view not in frames_by_view for view in secondary):
        missing = [view for view in secondary if view not in frames_by_view]
        raise KeyError(f"Secondary views not found in frames_by_view: {missing}")
    if not 0 < primary_height_fraction < 1:
        raise ValueError("primary_height_fraction must be between 0 and 1")

    views = [frames_by_view[primary_view], *(frames_by_view[view] for view in secondary)]
    frame_count = min(len(view) for view in views)
    if frame_count == 0:
        raise ValueError("Cannot tile empty views")
    if any(view.ndim != 4 or view.shape[-1] != 3 for view in views):
        raise ValueError("All views must have shape (T, H, W, 3)")

    primary_height = int(target_height * primary_height_fraction)
    lower_height = target_height - primary_height
    columns = max(1, int(np.ceil(np.sqrt(len(secondary)))))
    rows = int(np.ceil(len(secondary) / columns))
    cell_width = target_width // columns
    cell_height = lower_height // rows
    output = np.full((frame_count, target_height, target_width, 3), fill_value, dtype=np.uint8)
    for index in range(frame_count):
        output[index, :primary_height] = cv2.resize(
            views[0][index], (target_width, primary_height), interpolation=cv2.INTER_AREA
        )
        for panel_index, view in enumerate(views[1:]):
            row, column = divmod(panel_index, columns)
            x_start = column * cell_width
            x_end = target_width if column == columns - 1 else (column + 1) * cell_width
            y_start = primary_height + row * cell_height
            y_end = target_height if row == rows - 1 else primary_height + (row + 1) * cell_height
            output[index, y_start:y_end, x_start:x_end] = cv2.resize(
                view[index], (x_end - x_start, y_end - y_start), interpolation=cv2.INTER_AREA
            )
    return output


def decode_video_to_frames(
    source: object,
    max_frames: int = 64,
    max_width: int | None = None,
) -> np.ndarray:
    """Decode a video into at most ``max_frames`` uniformly spaced RGB frames.

    ``source`` may be a filesystem path, a ``bytes`` blob, or a binary file-like
    object (e.g. an uploaded ``UploadFile.file``). Uses PyAV/libdav1d so both
    AV1 and H.264 inputs decode without hardware acceleration.

    Returns a ``(T, H, W, 3)`` uint8 RGB array. Raises ``ValueError`` for empty
    or undecodable inputs.
    """
    import io

    import av

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if hasattr(source, "read"):
        source.seek(0)

    with av.open(source) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        duration = float(stream.duration * stream.time_base) if stream.duration else None
        rate = float(stream.average_rate) if stream.average_rate else None
        total = stream.frames or (int(round(duration * rate)) if duration and rate else None)

        frames: list[np.ndarray] = []
        if duration and duration > 0:
            keep = max(1, min(max_frames, int(round(duration * rate)) if rate else max_frames))
            targets = np.linspace(0.0, duration, keep, endpoint=False)
            target_index = 0
            for frame in container.decode(stream):
                timestamp = frame.time
                if timestamp is None:
                    continue
                while target_index < keep and timestamp >= targets[target_index]:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    target_index += 1
                if target_index >= keep:
                    break
        else:
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) > max_frames:
                indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                frames = [frames[i] for i in indices]

        if not frames:
            raise ValueError("No frames decoded from input video")

        width, height = frames[0].shape[1], frames[0].shape[0]
        if total is not None and total <= 0:
            total = len(frames)

    array = np.stack(frames)
    if max_width and array.shape[2] > max_width:
        scale = max_width / array.shape[2]
        new_w = int(round(array.shape[2] * scale))
        new_h = int(round(array.shape[1] * scale))
        array = np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in array])
    return array


def views_from_video_files(
    video_files: dict[str, object],
    primary_view: str | None = None,
    secondary_views: list[str] | None = None,
    max_frames: int = 64,
    max_width: int | None = None,
) -> tuple[dict[str, np.ndarray], str, list[str]]:
    """Decode one video per view into a synchronized ``views`` mapping.

    Views are truncated to the shortest decoded length so all panels stay
    aligned. Returns ``(views, primary_view, secondary_views)``.
    """
    if not video_files:
        raise ValueError("No view videos provided")

    decoded = {name: decode_video_to_frames(src, max_frames, max_width) for name, src in video_files.items()}
    frame_count = min(len(frames) for frames in decoded.values())
    if frame_count < 2:
        raise ValueError(f"Need at least 2 synchronized frames, got {frame_count}")

    views = {name: frames[:frame_count] for name, frames in decoded.items()}
    names = list(views)
    primary = primary_view or next((n for n in ("top", "front", "agentview", "left") if n in names), names[0])
    if primary not in views:
        raise KeyError(f"primary_view {primary!r} not among views {names}")
    secondary = [n for n in (secondary_views or [n for n in names if n != primary]) if n != primary]
    if not secondary:
        raise ValueError("At least one secondary view is required for tiling")
    missing = [n for n in secondary if n not in views]
    if missing:
        raise ValueError(f"Secondary views not uploaded: {missing}; got {names}")
    return views, primary, secondary
