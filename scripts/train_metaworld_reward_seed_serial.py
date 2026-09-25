#!/usr/bin/env python3
"""Train the selected MetaWorld reward/seed matrix serially."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import List, NamedTuple, Optional, Sequence, Tuple

from train_metaworld_reward_serial import (
    CONDA_ENV,
    PROJECT_DIR,
    _install_signal_handlers,
    terminate_process_group,
)


# (Hydra task config, user-facing MetaWorld task name)
TASKS: Tuple[Tuple[str, str], ...] = (
    ("button-press-wall", "button-press-wall"),
    ("coffee-push", "coffee-push"),
    ("soccer-meta", "soccer"),
    ("drawer-open", "drawer-open"),
    ("sweep-into", "sweep-into"),
    ("assembly", "assembly"),
)
REWARD_TYPES: Tuple[str, ...] = ("sparse", "pbrs", "dense")
SEEDS: Tuple[int, ...] = (121, 122, 123, 124, 125)
DEFAULT_START_INDEX = 18


class Experiment(NamedTuple):
    task_config: str
    task_name: str
    reward_type: str
    seed: int


EXPERIMENTS: Tuple[Experiment, ...] = tuple(
    Experiment(task_config, task_name, reward_type, seed)
    for task_config, task_name in TASKS
    for reward_type in REWARD_TYPES
    for seed in SEEDS
)


def make_run_name(task_name: str, reward_type: str, seed: int) -> str:
    return f"{reward_type}-{task_name}-seed{seed}"


def reward_overrides(reward_type: str) -> Tuple[str, ...]:
    if reward_type == "sparse":
        return ("reward_type=sparse", "shaped_reward.enabled=false")
    if reward_type == "pbrs":
        return (
            "reward_type=sparse",
            "shaped_reward.enabled=true",
            "shaped_reward.type=pbrs",
        )
    if reward_type == "dense":
        return ("reward_type=dense", "shaped_reward.enabled=false")
    raise ValueError(f"Unsupported reward type: {reward_type}")


def build_command(experiment: Experiment) -> List[str]:
    """Build one training command without changing task-config parameters."""

    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "train_mw.py",
        f"task={experiment.task_config}",
        "agent=drm_mw",
    ]
    command.extend(reward_overrides(experiment.reward_type))
    command.extend(
        (
            f"seed={experiment.seed}",
            "wandb_run_name="
            + make_run_name(
                experiment.task_name,
                experiment.reward_type,
                experiment.seed,
            ),
        )
    )
    return command


def run_experiment(experiment: Experiment) -> int:
    command = build_command(experiment)
    print(f"Command: {shlex.join(command)}", flush=True)
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            start_new_session=True,
        )
    except OSError as error:
        print(f"Could not start experiment: {error}", file=sys.stderr)
        return 127

    try:
        return process.wait()
    finally:
        terminate_process_group(process)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Train {len(TASKS)} MetaWorld tasks with {len(REWARD_TYPES)} "
            f"reward types and {len(SEEDS)} seeds serially "
            f"({len(EXPERIMENTS)} experiments total)."
        )
    )
    parser.add_argument(
        "--start-index",
        type=int,
        choices=range(1, len(EXPERIMENTS) + 1),
        default=DEFAULT_START_INDEX,
        metavar=f"1-{len(EXPERIMENTS)}",
        help=(
            "1-based experiment index to start from "
            f"(default: {DEFAULT_START_INDEX})."
        ),
    )
    parser.add_argument(
        "--end-index",
        type=int,
        choices=range(1, len(EXPERIMENTS) + 1),
        default=len(EXPERIMENTS),
        metavar=f"1-{len(EXPERIMENTS)}",
        help=(
            "1-based experiment index to stop at, inclusive "
            f"(default: {len(EXPERIMENTS)})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the selected commands without starting training."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.end_index < args.start_index:
        raise ValueError("end-index must be greater than or equal to start-index")

    selected_experiments = EXPERIMENTS[args.start_index - 1:args.end_index]
    if args.dry_run:
        for experiment in selected_experiments:
            print(shlex.join(build_command(experiment)))
        print(
            f"Dry run: printed {len(selected_experiments)} commands "
            f"for experiments {args.start_index}-{args.end_index}; "
            "no experiments were started."
        )
        return 0

    _install_signal_handlers()
    failures: List[str] = []
    for index, experiment in enumerate(
        selected_experiments,
        start=args.start_index,
    ):
        run_name = make_run_name(
            experiment.task_name,
            experiment.reward_type,
            experiment.seed,
        )
        print(
            f"========== Starting experiment {index}/{len(EXPERIMENTS)}: "
            f"{run_name} ==========",
            flush=True,
        )
        status = run_experiment(experiment)
        if status == 0:
            print(
                f"========== Finished experiment {index}/{len(EXPERIMENTS)}: "
                f"{run_name} ==========",
                flush=True,
            )
        else:
            failures.append(f"{run_name} (exit={status})")
            print(
                f"========== Experiment failed with exit={status}; "
                "continuing ==========",
                file=sys.stderr,
                flush=True,
            )

    if failures:
        print(
            "Completed with failed experiments: " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    print(
        f"All {len(selected_experiments)} selected MetaWorld experiments "
        "finished successfully."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
