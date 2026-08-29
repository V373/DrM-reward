#!/usr/bin/env python3
"""Run the selected MetaWorld tasks serially with sparse and dense rewards."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONDA_ENV = "drm-cu128"

# The names on the right are the Hydra task-config names used by this project.
# In particular, the existing configs are called soccer-meta, hammer-meta, and
# disassemble even though the corresponding MetaWorld tasks are soccer, hammer,
# and disassembly.
TASKS: Tuple[Tuple[str, str], ...] = (
    ("button press wall", "button-press-wall"),
    ("coffee push", "coffee-push"),
    ("soccer", "soccer-meta"),
    ("window close", "window-close"),
    ("drawer open", "drawer-open"),
    ("sweep into", "sweep-into"),
    ("door lock", "door-lock"),
    ("hammer", "hammer-meta"),
    ("assembly", "assembly"),
    ("disassembly", "disassemble"),
)
REWARD_TYPES: Tuple[str, ...] = ("sparse", "dense")


def wait_for_process_group(process_group: int) -> None:
    """Wait until all descendants in an experiment's process group exit."""

    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            # The group belongs to the child process started by this script;
            # this should not normally happen, but it is safer to continue
            # than to wait forever if the OS denies the probe.
            return
        time.sleep(1)


def build_command(task_config: str, reward_type: str) -> List[str]:
    """Build the same training command used by train_metaworld_serial.sh."""

    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "train_mw.py",
        f"task={task_config}",
        "agent=drm_mw",
        f"reward_type={reward_type}",
        f"wandb_run_name_prefix={reward_type.upper()}-",
    ]


def run_experiment(
    task_config: str,
    reward_type: str,
    dry_run: bool = False,
) -> int:
    command = build_command(task_config, reward_type)
    print(f"Command: {shlex.join(command)}", flush=True)

    if dry_run:
        return 0

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            start_new_session=True,
        )
    except OSError as error:
        print(f"Could not start experiment: {error}", file=sys.stderr)
        return 127

    # start_new_session=True creates a new session whose process group ID is
    # the child's PID.  This also covers workers that outlive conda's launcher.
    process_group = process.pid
    return_code = process.wait()
    wait_for_process_group(process_group)
    return return_code


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 10 MetaWorld tasks serially with both sparse and dense "
            "rewards (20 experiments total)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all 20 commands without starting training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    failed_experiments: List[str] = []
    total = len(TASKS) * len(REWARD_TYPES)
    experiment_number = 0

    for reward_type in REWARD_TYPES:
        for display_name, task_config in TASKS:
            experiment_number += 1
            prefix = f"{reward_type.upper()}-"
            label = f"{task_config} ({display_name}, {reward_type})"
            print(
                f"========== Starting experiment "
                f"{experiment_number}/{total}: {label}; W&B prefix={prefix} "
                f"==========",
                flush=True,
            )

            status = run_experiment(task_config, reward_type, args.dry_run)
            if status == 0:
                print(
                    f"========== Finished experiment "
                    f"{experiment_number}/{total}: {label} ==========",
                    flush=True,
                )
            else:
                failed_experiments.append(f"{label}(exit={status})")
                print(
                    f"========== Experiment failed with exit={status}; "
                    "continuing ==========",
                    flush=True,
                )

    if failed_experiments:
        print(
            "Completed with failed experiments: "
            + ", ".join(failed_experiments),
            file=sys.stderr,
        )
        return 1

    print("All 20 MetaWorld experiments finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
