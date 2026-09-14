import sys
from pathlib import Path

import pytest
import yaml


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from train_metaworld_reward_seed_serial import (  # noqa: E402
    EXPERIMENTS,
    REWARD_TYPES,
    SEEDS,
    TASKS,
    Experiment,
    build_command,
    main,
    make_run_name,
)


def test_experiment_matrix_is_complete_unique_and_task_grouped():
    expected = [
        Experiment(task_config, task_name, reward_type, seed)
        for task_config, task_name in TASKS
        for reward_type in REWARD_TYPES
        for seed in SEEDS
    ]

    assert len(EXPERIMENTS) == 90
    assert len(set(EXPERIMENTS)) == 90
    assert list(EXPERIMENTS) == expected


@pytest.mark.parametrize(
    ("reward_type", "expected_overrides", "unexpected_overrides"),
    [
        (
            "sparse",
            {"reward_type=sparse", "shaped_reward.enabled=false"},
            {"reward_type=dense", "shaped_reward.type=pbrs"},
        ),
        (
            "pbrs",
            {
                "reward_type=sparse",
                "shaped_reward.enabled=true",
                "shaped_reward.type=pbrs",
            },
            {"reward_type=dense", "shaped_reward.enabled=false"},
        ),
        (
            "dense",
            {"reward_type=dense", "shaped_reward.enabled=false"},
            {"reward_type=sparse", "shaped_reward.type=pbrs"},
        ),
    ],
)
def test_build_command_selects_reward_without_training_parameter_overrides(
    reward_type, expected_overrides, unexpected_overrides
):
    experiment = Experiment("soccer-meta", "soccer", reward_type, 123)

    command = build_command(experiment)

    assert "task=soccer-meta" in command
    assert "agent=drm_mw" in command
    assert "seed=123" in command
    assert f"wandb_run_name={reward_type}-soccer-seed123" in command
    assert expected_overrides <= set(command)
    assert unexpected_overrides.isdisjoint(command)
    assert not any(argument.startswith("discount=") for argument in command)
    assert not any(argument.startswith("nstep=") for argument in command)


def test_all_run_names_are_exact_and_unique():
    names = [
        make_run_name(experiment.task_name, experiment.reward_type, experiment.seed)
        for experiment in EXPERIMENTS
    ]

    assert len(set(names)) == 90
    assert names[0] == "sparse-button-press-wall-seed121"
    assert names[-1] == "dense-assembly-seed125"


def test_dry_run_starts_at_default_experiment(capsys):
    assert main(["--dry-run"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 74
    assert lines[0].endswith(
        "seed=123 wandb_run_name=sparse-coffee-push-seed123"
    )
    assert lines[-2].endswith(
        "seed=125 wandb_run_name=dense-assembly-seed125"
    )
    assert lines[-1] == (
        "Dry run: printed 73 commands starting at experiment 18; "
        "no experiments were started."
    )


def test_start_index_can_include_the_full_matrix(capsys):
    assert main(["--dry-run", "--start-index", "1"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 91
    assert lines[0].endswith(
        "seed=121 wandb_run_name=sparse-button-press-wall-seed121"
    )
    assert lines[-1] == (
        "Dry run: printed 90 commands starting at experiment 1; "
        "no experiments were started."
    )


def test_full_wandb_run_name_is_declared_in_local_hydra_config():
    config_path = Path(__file__).resolve().parents[1] / "cfgs" / "config.yaml"

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["wandb_run_name"] == ""
