#!/usr/bin/env python3
"""Convert MetaWorld HDF5 RGB observation trajectories to MP4 videos.

The videos are written to ``<hdf5-directory>/video/<hdf5-stem>/traj_N.mp4``.
The FPS is read from ``env_metadata.fps``, which is saved by
``training_data_gen.py``; it defaults to 80 only for legacy files that lack
this metadata.

Examples:
    # Convert every HDF5 file in DrM/metaworld/data (the default input).
    python metaworld/data/hdf5_obs_to_mp4.py

    # Convert one dataset, replacing videos that already exist.
    python metaworld/data/hdf5_obs_to_mp4.py data/example.hdf5 --overwrite

    # Also write crop-box previews while retaining the normal MP4 files.
    python metaworld/data/hdf5_obs_to_mp4.py data/example.hdf5 \
        --center-crop 112 112 --crop-offset 56 112
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import h5py
import imageio
import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_FPS = 80
CENTER_CROP_BOX_COLOR = np.array([255, 0, 0], dtype=np.uint8)
CENTER_CROP_BOX_THICKNESS = 2


def trajectory_sort_key(name: str) -> tuple:
    """Put traj_2 before traj_10 while retaining a stable fallback."""
    suffix = name.removeprefix("traj_")
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, suffix)


def find_hdf5_files(inputs: Iterable[Path]) -> List[Path]:
    """Expand HDF5 files and non-recursive directories into sorted file paths."""
    paths = set()
    for input_path in inputs:
        input_path = input_path.expanduser()
        if input_path.is_file():
            if input_path.suffix.lower() not in {".h5", ".hdf5"}:
                raise ValueError(f"Not an HDF5 file: {input_path}")
            paths.add(input_path.resolve())
        elif input_path.is_dir():
            paths.update(path.resolve() for path in input_path.glob("*.h5"))
            paths.update(path.resolve() for path in input_path.glob("*.hdf5"))
        else:
            raise FileNotFoundError(f"Input does not exist: {input_path}")
    return sorted(paths)


def read_fps(datafile: h5py.File, fallback_fps: int) -> int:
    """Read the generator's saved FPS, falling back for old datasets."""
    raw_metadata = datafile.attrs.get("env_metadata")
    if raw_metadata is not None:
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        try:
            fps = json.loads(raw_metadata).get("fps")
            if fps is not None and float(fps) > 0:
                return int(round(float(fps)))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return fallback_fps


def crop_box_bounds(
    image_height: int,
    image_width: int,
    crop_height: int,
    crop_width: int,
    crop_offset: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    """Return (top, left, bottom, right) for a centered or translated crop."""
    if crop_offset is None:
        top = (image_height - crop_height) // 2
        left = (image_width - crop_width) // 2
    else:
        left, top = crop_offset
        if left < 0 or top < 0:
            raise ValueError(
                f"Crop offset must be non-negative, got left={left}, top={top}"
            )
    bottom = top + crop_height
    right = left + crop_width
    if bottom > image_height or right > image_width:
        location = "centered" if crop_offset is None else f"left={left}, top={top}"
        raise ValueError(
            f"Crop {crop_height}x{crop_width} at {location} exceeds frame size "
            f"{image_height}x{image_width}"
        )
    return top, left, bottom, right


def draw_crop_box(
    image: np.ndarray,
    crop_height: int,
    crop_width: int,
    crop_offset: tuple[int, int] | None,
) -> None:
    """Draw a red HxW crop box in-place on an RGB image."""
    image_height, image_width = image.shape[:2]
    top, left, bottom, right = crop_box_bounds(
        image_height, image_width, crop_height, crop_width, crop_offset
    )
    thickness = min(CENTER_CROP_BOX_THICKNESS, crop_height, crop_width)
    image[top : top + thickness, left:right] = CENTER_CROP_BOX_COLOR
    image[bottom - thickness : bottom, left:right] = CENTER_CROP_BOX_COLOR
    image[top:bottom, left : left + thickness] = CENTER_CROP_BOX_COLOR
    image[top:bottom, right - thickness : right] = CENTER_CROP_BOX_COLOR


def write_center_crop_preview(
    observations: h5py.Dataset,
    output_path: Path,
    crop_height: int,
    crop_width: int,
    crop_offset: tuple[int, int] | None,
) -> None:
    """Save the first RGB frame with its crop region marked in red."""
    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB observations with shape (T, H, W, 3), got "
            f"{observations.shape} in {observations.name}"
        )
    if observations.shape[0] == 0:
        raise ValueError(f"Cannot create crop preview for empty trajectory: {observations.parent.name}")

    frame = observations[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    image = frame[..., ::-1].copy()  # MWImgObs stores BGR frames.
    draw_crop_box(image, crop_height, crop_width, crop_offset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, image)
    location = "centered" if crop_offset is None else f"left={crop_offset[0]}, top={crop_offset[1]}"
    print(f"  wrote {output_path.name} (first frame with {crop_height}x{crop_width} crop at {location})")


def write_trajectory_video(
    observations: h5py.Dataset,
    output_path: Path,
    fps: int,
    overwrite: bool,
    crop_size: tuple[int, int] | None,
    crop_offset: tuple[int, int] | None,
) -> bool:
    """Stream one trajectory to MP4. Return False when an existing file is kept.

    ``crop_size`` is used only for the optional crop-box preview video; normal
    trajectory MP4s are always written without an overlay.
    """
    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB observations with shape (T, H, W, 3), got "
            f"{observations.shape} in {observations.name}"
        )
    if observations.shape[0] <= 1:
        print(f"  skip trajectory without frames after dropping first: {observations.parent.name}")
        return False
    if output_path.exists() and not overwrite:
        print(f"  exists, skipping: {output_path.name}")
        return False
    if crop_size is not None:
        crop_height, crop_width = crop_size
        frame_height, frame_width = observations.shape[1:3]
        crop_box_bounds(frame_height, frame_width, crop_height, crop_width, crop_offset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.mp4")
    if temporary_path.exists():
        temporary_path.unlink()

    # MWImgObs stores the offscreen MuJoCo image in BGR channel order
    # (`sim.render(...)[..., ::-1]`). Convert it to RGB, then use the same
    # H.264 (libx264/yuv420p) encoding defaults as VideoRecorder's
    # imageio.mimsave call used for W&B eval/video uploads.
    try:
        with imageio.get_writer(
            temporary_path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        ) as writer:
            # The first saved observation is the reset-state frame; omit it
            # from exported rollout videos.
            for frame_index in range(1, observations.shape[0]):
                frame = observations[frame_index]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                image = frame[..., ::-1]
                if crop_size is not None:
                    image = image.copy()
                    draw_crop_box(image, *crop_size, crop_offset)
                writer.append_data(image)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"  wrote {output_path.name} ({observations.shape[0] - 1} frames, {fps} fps)")
    return True


def convert_file(
    hdf5_path: Path,
    output_dir: Path | None,
    trajectory_indices: set[int] | None,
    fallback_fps: int,
    overwrite: bool,
    crop_size: tuple[int, int] | None,
    crop_offset: tuple[int, int] | None,
) -> int:
    """Convert all selected trajectories in one HDF5 file and return count."""
    written = 0
    with h5py.File(hdf5_path, "r") as datafile:
        fps = read_fps(datafile, fallback_fps)
        trajectory_names = sorted(
            (name for name in datafile if name.startswith("traj_") and isinstance(datafile[name], h5py.Group)),
            key=trajectory_sort_key,
        )
        if not trajectory_names:
            raise ValueError(f"No traj_* groups found in {hdf5_path}")

        print(f"Converting {hdf5_path.name}: {len(trajectory_names)} trajectories at {fps} fps")
        destination_root = output_dir or hdf5_path.parent / "video"
        destination_dir = destination_root / hdf5_path.stem
        if crop_size is not None:
            first_trajectory = datafile[trajectory_names[0]]
            if "observations" not in first_trajectory:
                raise KeyError(f"Missing observations in {hdf5_path}:{trajectory_names[0]}")
            crop_height, crop_width = crop_size
            write_center_crop_preview(
                first_trajectory["observations"],
                destination_dir / f"first_frame_center_crop_{crop_height}x{crop_width}.png",
                crop_height,
                crop_width,
                crop_offset,
            )

        for trajectory_name in trajectory_names:
            try:
                trajectory_index = int(trajectory_name.removeprefix("traj_"))
            except ValueError:
                trajectory_index = None
            if trajectory_indices is not None and trajectory_index not in trajectory_indices:
                continue
            if "observations" not in datafile[trajectory_name]:
                raise KeyError(f"Missing observations in {hdf5_path}:{trajectory_name}")

            # Always retain a clean video for direct viewing.
            output_path = destination_dir / f"{trajectory_name}.mp4"
            written += write_trajectory_video(
                datafile[trajectory_name]["observations"],
                output_path,
                fps,
                overwrite,
                None,
                None,
            )
            if crop_size is not None:
                crop_preview_path = destination_dir / f"{trajectory_name}_crop_box.mp4"
                written += write_trajectory_video(
                    datafile[trajectory_name]["observations"],
                    crop_preview_path,
                    fps,
                    overwrite,
                    crop_size,
                    crop_offset,
                )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[DEFAULT_DATA_DIR],
        help=f"HDF5 files or directories to convert (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional video root directory. Each HDF5 still gets its own "
            "<hdf5-stem>/ subdirectory (default: <hdf5-directory>/video)"
        ),
    )
    parser.add_argument(
        "--traj",
        type=int,
        nargs="+",
        help="Only convert these numeric trajectory indices, e.g. --traj 0 3 10",
    )
    parser.add_argument(
        "--fallback-fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"FPS for legacy files without env_metadata.fps (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--center-crop",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help=(
            "Also write a crop-box PNG and *_crop_box.mp4 preview at HxW; "
            "the normal MP4s remain unmarked. Centered unless --crop-offset "
            "is also specified"
        ),
    )
    parser.add_argument(
        "--crop-offset",
        type=int,
        nargs=2,
        metavar=("LEFT", "TOP"),
        help=(
            "Place the crop box with its left and top edges LEFT and TOP pixels "
            "from the image borders; requires --center-crop H W"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace MP4 files that already exist")
    args = parser.parse_args()
    if args.fallback_fps <= 0:
        parser.error("--fallback-fps must be positive")
    if args.center_crop is not None and any(size <= 0 for size in args.center_crop):
        parser.error("--center-crop H W must both be positive")
    if args.crop_offset is not None and args.center_crop is None:
        parser.error("--crop-offset requires --center-crop H W")
    if args.crop_offset is not None and any(gap < 0 for gap in args.crop_offset):
        parser.error("--crop-offset LEFT TOP must both be non-negative")
    return args


def main() -> None:
    args = parse_args()
    hdf5_paths = find_hdf5_files(args.inputs)
    if not hdf5_paths:
        raise FileNotFoundError("No .h5 or .hdf5 files found in the requested input")

    total_written = 0
    trajectory_indices = set(args.traj) if args.traj is not None else None
    for hdf5_path in hdf5_paths:
        total_written += convert_file(
            hdf5_path,
            args.output_dir.resolve() if args.output_dir else None,
            trajectory_indices,
            args.fallback_fps,
            args.overwrite,
            tuple(args.center_crop) if args.center_crop is not None else None,
            tuple(args.crop_offset) if args.crop_offset is not None else None,
        )
    print(f"Done: wrote {total_written} MP4 file(s).")


if __name__ == "__main__":
    main()
