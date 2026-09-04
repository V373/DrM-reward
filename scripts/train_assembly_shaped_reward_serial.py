#!/usr/bin/env python3
"""Train Assembly PBRS n-step ablations serially."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import Optional, Sequence

from train_metaworld_reward_serial import (
    CONDA_ENV,
    PROJECT_DIR,
    _install_signal_handlers,
    terminate_process_group,
)

EXPERIMENTS = (
    # ("pbrs", "SHAPED_PBRS-NSTEP1-", 1, None),
    ("pbrs", "SHAPED_PBRS-NSTEP3-DISCOUNT0.995-", 3, 0.995),
    # ("pbrs", "SHAPED_PBRS-NSTEP3-DISCOUNT0.99-", 3, 0.99),
    # ("pbrs", "SHAPED_PBRS-NSTEP3-DISCOUNT0.98-", 3, 0.98),
)


def build_command(
    shaped_reward_type: str,
    run_name_prefix: str,
    nstep: int,
    discount: Optional[float] = None,
) -> list[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "train_mw.py",
        "task=assembly",
        "agent=drm_mw",
        "reward_type=sparse",
        f"shaped_reward.type={shaped_reward_type}",
        f"nstep={nstep}",
        "nstep_alpha=0",
    ]
    if discount is not None:
        command.extend([
            f"discount={discount}",
            "discount_alpha=0",
            "discount_beta=0",
            f"shaped_reward.pbrs_gamma={discount}",
        ])
    command.append(f"wandb_run_name_prefix={run_name_prefix}")
    return command


def run_experiment(
    shaped_reward_type: str,
    run_name_prefix: str,
    nstep: int,
    discount: Optional[float] = None,
) -> int:
    command = build_command(
        shaped_reward_type, run_name_prefix, nstep, discount
    )
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
        description="Train Assembly PBRS n-step ablations serially."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without starting training.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        for shaped_reward_type, run_name_prefix, nstep, discount in EXPERIMENTS:
            print(
                shlex.join(
                    build_command(
                        shaped_reward_type, run_name_prefix, nstep, discount
                    )
                )
            )
        return 0

    _install_signal_handlers()

    failures = []
    for index, (
        shaped_reward_type,
        run_name_prefix,
        nstep,
        discount,
    ) in enumerate(EXPERIMENTS, start=1):
        discount_label = "" if discount is None else f", discount={discount}"
        label = f"assembly ({shaped_reward_type}, nstep={nstep}{discount_label})"
        print(f"Starting {index}/{len(EXPERIMENTS)}: {label}", flush=True)
        status = run_experiment(
            shaped_reward_type, run_name_prefix, nstep, discount
        )
        if status != 0:
            failures.append(f"{label} (exit={status})")
            print(f"Failed: {failures[-1]}; continuing.", file=sys.stderr)
        else:
            print(f"Finished {index}/{len(EXPERIMENTS)}: {label}", flush=True)

    if failures:
        print("Completed with failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("All Assembly shaped-reward experiments finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
