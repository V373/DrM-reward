#!/usr/bin/env python3
"""Run the selected MetaWorld tasks serially with sparse and dense rewards."""

from __future__ import annotations

import argparse
import os
import signal
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONDA_ENV = "drm-cu128"

# Hydra task-config names used by this project.
TASKS: Tuple[str, ...] = (
    "button-press-wall",
    "coffee-push",
    "soccer-meta",
    "window-close",
    "drawer-open",
    "sweep-into",
    "door-lock",
    "hammer-meta",
    "assembly",
    "disassemble",
)
REWARD_TYPES: Tuple[str, ...] = ("sparse", "dense")

TERM_TIMEOUT_SECONDS = 10.0
KILL_TIMEOUT_SECONDS = 5.0
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def wait_for_process_group(
    process: subprocess.Popen,
    timeout: float,
) -> bool:
    """Return whether the experiment leader and its process group exited."""

    deadline = time.monotonic() + timeout
    if process.poll() is None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False

    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.2, remaining))


def terminate_process_group(process: subprocess.Popen) -> None:
    """Stop an experiment and its workers, escalating if necessary."""

    for signum, timeout in (
        (signal.SIGTERM, TERM_TIMEOUT_SECONDS),
        (signal.SIGKILL, KILL_TIMEOUT_SECONDS),
    ):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            process.poll()  # Reap the leader if it exited between checks.
            return
        except PermissionError:
            print(
                f"Could not signal process group {process.pid}: permission denied",
                file=sys.stderr,
                flush=True,
            )
            return

        if wait_for_process_group(process, timeout):
            return
        if signum == signal.SIGTERM:
            print(
                f"Process group {process.pid} did not exit after SIGTERM; "
                "sending SIGKILL.",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"Process group {process.pid} is still present after SIGKILL.",
        file=sys.stderr,
        flush=True,
    )


def _handle_stop_signal(signum: int, _frame) -> None:
    """Abort the loop; run_experiment's finally block cleans up workers."""

    for stop_signal in STOP_SIGNALS:
        signal.signal(stop_signal, signal.SIG_DFL)
    print(
        f"Received {signal.Signals(signum).name}; stopping.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(128 + signum)


def _install_signal_handlers() -> None:
    for signum in STOP_SIGNALS:
        signal.signal(signum, _handle_stop_signal)


def build_command(task_config: str, reward_type: str) -> List[str]:
    """Build one MetaWorld training command."""

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
) -> int:
    command = build_command(task_config, reward_type)
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

    # A separate session lets cleanup address conda, train_mw.py, and all
    # multiprocessing workers as one process group.
    try:
        return process.wait()
    finally:
        terminate_process_group(process)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    total = len(TASKS) * len(REWARD_TYPES)
    parser = argparse.ArgumentParser(
        description=(
            f"Run {len(TASKS)} MetaWorld tasks serially with both sparse and "
            f"dense rewards ({total} experiments total)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Print all {total} commands without starting training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    experiments = [
        (task_config, reward_type)
        for reward_type in REWARD_TYPES
        for task_config in TASKS
    ]
    total = len(experiments)

    if args.dry_run:
        for task_config, reward_type in experiments:
            print(shlex.join(build_command(task_config, reward_type)))
        print(f"Dry run: printed {total} commands; no experiments were started.")
        return 0

    _install_signal_handlers()
    failed_experiments: List[str] = []

    for experiment_number, (task_config, reward_type) in enumerate(
        experiments,
        start=1,
    ):
        label = f"{task_config} ({reward_type})"
        print(
            f"========== Starting experiment "
            f"{experiment_number}/{total}: {label} ==========",
            flush=True,
        )

        status = run_experiment(task_config, reward_type)
        if status == 0:
            print(
                f"========== Finished experiment "
                f"{experiment_number}/{total}: {label} ==========",
                flush=True,
            )
        else:
            failed_experiments.append(f"{label} (exit={status})")
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

    print(f"All {total} MetaWorld experiments finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
