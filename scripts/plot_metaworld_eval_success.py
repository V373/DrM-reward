#!/usr/bin/env python3
"""Plot MetaWorld evaluation success-rate curves from selected local runs.

The experiment paths are deliberately explicit: this prevents historical,
failed, and duplicate runs under ``exp_local`` from being included by mistake.

Run this script in the isolated plotting environment:
    conda run -n drm-plot python scripts/plot_metaworld_eval_success.py
"""

from __future__ import annotations

import csv
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit(
        "Matplotlib is required. Install it with: "
        "conda create -n drm-plot python=3.11 numpy=1.26 matplotlib-base"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
EXP_ROOT = PROJECT_DIR / "exp_local"
IMAGE_ROOT = PROJECT_DIR / "images" / "metaworld_eval_success"

SEEDS: Tuple[int, ...] = (121, 122, 123, 124, 125)
REWARDS: Tuple[str, ...] = ("sparse", "pbrs", "dense")
MIN_COMPLETE_EVAL_FRAME = 2_090_000
T_CRITICAL_95_DF4 = 2.776445
DEFAULT_DOWNSAMPLE = 5

# Edit these two constants to select a different fully-completed task set.
# Each reward list must be ordered by SEEDS (121, 122, 123, 124, 125).
SELECTED_TASKS: Tuple[str, ...] = (
    "sweep-into",
    "assembly",
)

RUN_DIRS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "sweep-into": {
        "sparse": (
            "2026.09.17/193750_sweep-into",
            "2026.09.17/214254_sweep-into",
            "2026.09.18/002906_sweep-into",
            "2026.09.18/023528_sweep-into",
            "2026.09.18/044115_sweep-into",
        ),
        "pbrs": (
            "2026.09.18/064914_sweep-into",
            "2026.09.18/094813_sweep-into",
            "2026.09.18/124807_sweep-into",
            "2026.09.18/154851_sweep-into",
            "2026.09.18/185159_sweep-into",
        ),
        "dense": (
            "2026.09.18/215303_sweep-into",
            "2026.09.19/000111_sweep-into",
            "2026.09.19/020850_sweep-into",
            "2026.09.19/041647_sweep-into",
            "2026.09.19/062430_sweep-into",
        ),
    },
    "assembly": {
        "sparse": (
            "2026.09.19/083255_assembly",
            "2026.09.19/103950_assembly",
            "2026.09.19/124901_assembly",
            "2026.09.19/145647_assembly",
            "2026.09.19/170428_assembly",
        ),
        "pbrs": (
            "2026.09.19/191036_assembly",
            "2026.09.19/220823_assembly",
            "2026.09.20/010509_assembly",
            "2026.09.20/040250_assembly",
            "2026.09.20/065952_assembly",
        ),
        "dense": (
            "2026.09.20/095635_assembly",
            "2026.09.20/120316_assembly",
            "2026.09.20/140925_assembly",
            "2026.09.20/161610_assembly",
            "2026.09.20/182406_assembly",
        ),
    },
    "drawer-open": {
        "sparse": (
            "2026.09.16/081941_drawer-open",
            "2026.09.16/102226_drawer-open",
            "2026.09.16/122450_drawer-open",
            "2026.09.16/142739_drawer-open",
            "2026.09.16/163101_drawer-open",
        ),
        "pbrs": (
            "2026.09.16/183237_drawer-open",
            "2026.09.16/212836_drawer-open",
            "2026.09.17/002517_drawer-open",
            "2026.09.17/032243_drawer-open",
            "2026.09.17/061916_drawer-open",
        ),
        "dense": (
            "2026.09.17/091530_drawer-open",
            "2026.09.17/111945_drawer-open",
            "2026.09.17/132453_drawer-open",
            "2026.09.17/153108_drawer-open",
            "2026.09.17/173504_drawer-open",
        ),
    },
    "button-press-wall": {
        "sparse": (
            "2026.09.10/182851_button-press-wall",
            "2026.09.10/204041_button-press-wall",
            "2026.09.10/225054_button-press-wall",
            "2026.09.11/010324_button-press-wall",
            "2026.09.11/031437_button-press-wall",
        ),
        "pbrs": (
            "2026.09.11/052708_button-press-wall",
            "2026.09.11/084236_button-press-wall",
            "2026.09.11/145011_button-press-wall",
            "2026.09.11/180449_button-press-wall",
            "2026.09.12/002457_button-press-wall",
        ),
        "dense": (
            "2026.09.12/034019_button-press-wall",
            "2026.09.12/054946_button-press-wall",
            "2026.09.12/075828_button-press-wall",
            "2026.09.12/100754_button-press-wall",
            "2026.09.12/121811_button-press-wall",
        ),
    },
    "coffee-push": {
        "sparse": (
            "2026.09.12/142747_coffee-push",
            "2026.09.12/163851_coffee-push",
            "2026.09.12/204640_coffee-push",
            "2026.09.12/225634_coffee-push",
            "2026.09.13/010513_coffee-push",
        ),
        "pbrs": (
            "2026.09.13/031601_coffee-push",
            "2026.09.13/062400_coffee-push",
            "2026.09.13/093355_coffee-push",
            "2026.09.13/124328_coffee-push",
            "2026.09.13/155443_coffee-push",
        ),
        "dense": (
            "2026.09.13/190458_coffee-push",
            "2026.09.13/211657_coffee-push",
            "2026.09.13/232913_coffee-push",
            "2026.09.14/014131_coffee-push",
            "2026.09.14/035222_coffee-push",
        ),
    },
}

REWARD_STYLES = {
    "sparse": ("Sparse", "#1f77b4"),
    "pbrs": ("PBRS", "#ff7f0e"),
    "dense": ("Dense", "#2ca02c"),
}


class DataError(RuntimeError):
    """Raised when a selected run cannot safely be used for plotting."""


def downsample_eval_success(
    values: Dict[int, float],
    factor: int,
) -> Dict[int, float]:
    """Keep every ``factor``-th evaluation point and always keep the last."""

    if factor < 1:
        raise ValueError(f"downsample factor must be >= 1, got {factor}")
    if factor == 1:
        return values

    frames = sorted(values)
    selected_frames = frames[::factor]
    if selected_frames[-1] != frames[-1]:
        selected_frames.append(frames[-1])
    return {frame: values[frame] for frame in selected_frames}


def load_eval_success(csv_path: Path, downsample: int) -> Dict[int, float]:
    """Read one eval.csv as a frame-to-success-rate mapping."""

    if not csv_path.is_file():
        raise DataError(f"missing eval.csv: {csv_path}")

    values: Dict[int, float] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"frame", "episode_success_rate"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise DataError(
                f"{csv_path} must contain columns: "
                "frame, episode_success_rate"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                frame = int(float(row["frame"]))
                success_rate = float(row["episode_success_rate"])
            except (TypeError, ValueError) as error:
                raise DataError(
                    f"invalid evaluation row at {csv_path}:{line_number}"
                ) from error
            if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
                raise DataError(
                    f"invalid success rate at {csv_path}:{line_number}: "
                    f"{success_rate}"
                )
            values[frame] = success_rate

    if not values:
        raise DataError(f"no evaluation rows in {csv_path}")
    if max(values) < MIN_COMPLETE_EVAL_FRAME:
        raise DataError(
            f"incomplete run ({max(values):,} < "
            f"{MIN_COMPLETE_EVAL_FRAME:,} eval frames): {csv_path.parent}"
        )
    return downsample_eval_success(values, downsample)


def summarize_seed_curves(
    seed_curves: Sequence[Dict[int, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return common frames, mean success rate, and a 95% t interval."""

    common_frames = set(seed_curves[0])
    for curve in seed_curves[1:]:
        common_frames.intersection_update(curve)
    if not common_frames:
        raise DataError("the five seeds have no common evaluation frames")

    frames = np.asarray(sorted(common_frames), dtype=float)
    rates = np.asarray(
        [[curve[int(frame)] for frame in frames] for curve in seed_curves],
        dtype=float,
    )
    mean = rates.mean(axis=0)
    sem = rates.std(axis=0, ddof=1) / math.sqrt(len(seed_curves))
    half_interval = T_CRITICAL_95_DF4 * sem
    lower = np.clip(mean - half_interval, 0.0, 1.0)
    upper = np.clip(mean + half_interval, 0.0, 1.0)
    return frames, mean, lower, upper


def collect_task_curves(
    task: str,
    downsample: int,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Validate and collect all 15 selected eval curves for one task."""

    if task not in RUN_DIRS:
        raise DataError(f"no hard-coded run mapping for task: {task}")

    result: Dict[str, Tuple[np.ndarray, ...]] = {}
    for reward in REWARDS:
        run_dirs = RUN_DIRS[task].get(reward)
        if run_dirs is None or len(run_dirs) != len(SEEDS):
            count = 0 if run_dirs is None else len(run_dirs)
            raise DataError(
                f"{task}/{reward} needs exactly {len(SEEDS)} seed directories "
                f"but has {count}"
            )
        seed_curves = [
            load_eval_success(EXP_ROOT / run_dir / "eval.csv", downsample)
            for run_dir in run_dirs
        ]
        result[reward] = summarize_seed_curves(seed_curves)
    return result


def plot_task(task: str, downsample: int) -> Tuple[Path, int]:
    """Create one success-rate plot containing the three reward variants."""

    curves = collect_task_curves(task, downsample)
    figure, axis = plt.subplots(figsize=(6.0, 4.0), dpi=300)
    for reward in REWARDS:
        frames, mean, lower, upper = curves[reward]
        label, color = REWARD_STYLES[reward]
        frames_in_millions = frames / 1_000_000.0
        line_width = 2.0 if reward == "pbrs" else 1.0
        axis.plot(frames_in_millions, mean, color=color, linewidth=line_width,
                  label=label)
        axis.fill_between(frames_in_millions, lower, upper, color=color,
                          alpha=0.20)

    axis.set_title(task)
    axis.set_xlabel("Env Steps (1M)")
    axis.set_ylabel("Success rate")
    axis.set_ylim(-0.1, 1.1)
    axis.margins(x=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Reward")
    figure.tight_layout()

    output_dir = IMAGE_ROOT / task
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_success_rate.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    point_count = len(curves[REWARDS[0]][0])
    return output_path, point_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected MetaWorld eval success-rate curves."
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
        metavar="N",
        help=(
            "Keep every Nth eval.csv point per seed while retaining the "
            f"last point (default: {DEFAULT_DOWNSAMPLE})."
        ),
    )
    args = parser.parse_args(argv)
    if args.downsample < 1:
        parser.error("--downsample must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated = 0
    for task in SELECTED_TASKS:
        try:
            output_path, point_count = plot_task(task, args.downsample)
        except DataError as error:
            print(f"[skip] {task}: {error}", file=sys.stderr)
            continue
        generated += 1
        print(
            f"[saved] {output_path} "
            f"({point_count} points, downsample={args.downsample})"
        )

    if generated == 0:
        print("No plots were generated.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
