from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pytest
import torch
import yaml

from replay_buffer import ReplayBuffer
from shaped_reward.gaussian_progress import BatchedGaussianProgressGatedProvider
from shaped_reward.metaworld import (
    MetaWorldShapedRewardManager,
    MetaWorldShapedRewardWrapper,
)
import shaped_reward.metaworld as shaped_metaworld


class _ProviderPlaceholder:
    def __init__(self, *args, **kwargs):
        self.progress_current = None


class _ScriptedProvider:
    def __init__(self, current, next_progress):
        self.progress_current = np.array([current], dtype=np.float64)
        self.next_progress = float(next_progress)

    def advance_all(self, frames, reset_mask=None):
        self.progress_current = np.array(
            [self.next_progress], dtype=np.float64)
        return self.progress_current


def _manager(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(
        shaped_metaworld,
        "BatchedGaussianProgressGatedProvider",
        _ProviderPlaceholder,
    )
    paths = {}
    for name in (
            "checkpoint_path",
            "gaussian_model_h5_path",
            "calibration_h5_path"):
        path = tmp_path / name
        path.touch()
        paths[name] = str(path)
    cfg = {
        "type": "pbrs",
        "sparse_scale": 1.0,
        "pbrs_gamma": 0.97,
        "pbrs_dense_scale": 1.0,
        "pbrs_progress_bias_enabled": False,
        "pbrs_progress_exp_enabled": False,
        "ood_p_value_threshold": 0.05,
        **paths,
        **overrides,
    }
    return MetaWorldShapedRewardManager(cfg, "cpu")


def test_dense_reward_uses_current_progress(tmp_path, monkeypatch):
    manager = _manager(
        tmp_path,
        monkeypatch,
        type="dense",
        sparse_scale=2.0,
        pbrs_dense_scale=3.0,
    )
    manager.provider = _ScriptedProvider(current=0.25, next_progress=0.75)

    reward, metrics = manager.step(1.0, np.empty(0), done=False)

    assert reward == pytest.approx(2.0 + 3.0 * 0.25)
    assert metrics["shaped_reward/progress"] == pytest.approx(0.25)
    assert metrics["shaped_reward/progress_next"] == 0.0


@pytest.mark.parametrize(
    ("done", "expected_shaping"),
    [
        (False, 0.97 * 0.7 - 0.2),
        (True, -0.2),
    ],
)
def test_pbrs_reward_and_terminal_potential(
        tmp_path, monkeypatch, done, expected_shaping):
    manager = _manager(
        tmp_path,
        monkeypatch,
        sparse_scale=2.0,
        pbrs_dense_scale=1.5,
    )
    manager.provider = _ScriptedProvider(current=0.2, next_progress=0.7)

    reward, metrics = manager.step(0.5, np.empty(0), done=done)

    assert reward == pytest.approx(1.0 + 1.5 * expected_shaping)
    assert metrics["shaped_reward/progress_next"] == pytest.approx(
        0.0 if done else 0.7)


def test_pbrs_bias_and_exponential_potential(tmp_path, monkeypatch):
    manager = _manager(
        tmp_path,
        monkeypatch,
        pbrs_gamma=0.5,
        pbrs_progress_bias_enabled=True,
        pbrs_progress_bias_b=0.1,
        pbrs_progress_exp_enabled=True,
        pbrs_progress_exp_base=2.0,
    )
    manager.provider = _ScriptedProvider(current=0.2, next_progress=0.4)

    reward, _ = manager.step(0.0, np.empty(0), done=False)

    expected = 0.5 * (2.0 ** 0.2) - 1.0
    assert reward == pytest.approx(expected)


class _FakeTimeStep(NamedTuple):
    observation: np.ndarray
    reward: float
    terminal: bool
    tag: str

    def last(self):
        return self.terminal


class _FakeEnv:
    def __init__(self):
        self.frame_value = 10

    def reset(self):
        self.frame_value = 10
        return _FakeTimeStep(np.array([1]), 0.0, False, "reset")

    def step(self, action):
        self.frame_value = 20
        return _FakeTimeStep(np.array([2]), 0.5, True, "step")

    def get_reward_frame(self, camera_name=None):
        assert camera_name == "reward_cam"
        return np.full((224, 224, 3), self.frame_value, dtype=np.uint8)


class _FakeManager:
    def __init__(self):
        self.reset_frame = None
        self.step_args = None

    def reset(self, frame):
        self.reset_frame = frame.copy()
        return 0.0

    def step(self, sparse_reward, next_frame, done):
        self.step_args = (sparse_reward, next_frame.copy(), done)
        return 9.0, {"shaped_reward/total": 9.0}


def test_wrapper_resets_progress_and_replaces_only_reward():
    env = _FakeEnv()
    manager = _FakeManager()
    wrapper = MetaWorldShapedRewardWrapper(
        env, {"reward_camera": "reward_cam"}, "cpu", manager=manager)

    first = wrapper.reset()
    step = wrapper.step(np.array([0.0]))

    assert first.tag == "reset"
    assert np.all(manager.reset_frame == 10)
    assert step.reward == 9.0
    assert step.observation.tolist() == [2]
    assert step.terminal and step.tag == "step"
    sparse_reward, next_frame, done = manager.step_args
    assert sparse_reward == 0.5 and done
    assert np.all(next_frame == 20)
    assert wrapper.last_reward_metrics == {"shaped_reward/total": 9.0}


class _RecordingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def forward(self, context):
        self.contexts.append(
            context[:, 0, :, 0, 0, 0].detach().cpu().numpy().copy())
        newest = context[:, 0, -1, 0, 0, 0]
        return newest.reshape(-1, 1, 1)


def _write_gaussian_assets(tmp_path):
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.touch()
    gaussian = tmp_path / "gaussian.h5"
    covariance = np.array([[[0.01]], [[0.01]]], dtype=np.float64)
    with h5py.File(gaussian, "w") as h5_file:
        h5_file.attrs["embedding_normalization"] = "none"
        h5_file.attrs["enable_pca"] = False
        h5_file.attrs["embedding_dim"] = 1
        h5_file.attrs["input_embedding_dim"] = 1
        model = h5_file.create_group("model")
        model.create_dataset("bin_progress_values", data=[0.0, 1.0])
        model.create_dataset("bin_means", data=[[0.0], [1.0]])
        model.create_dataset("bin_independent_covariances", data=covariance)
        model.create_dataset("shared_covariance", data=[[0.01]])
        model.create_dataset("bin_final_covariances", data=covariance)
        model.create_dataset(
            "bin_log_determinants", data=np.log([0.01, 0.01]))
        model.create_dataset("bin_counts", data=[4, 4])

    calibration = tmp_path / "calibration.h5"
    with h5py.File(calibration, "w") as h5_file:
        h5_file.attrs["embedding_normalization"] = "none"
        video = h5_file.create_group("videos").create_group("0")
        video.create_dataset(
            "embeddings",
            data=np.array([-.05, 0.0, .05, .08,
                           .92, .95, 1.0, 1.05])[:, None],
        )
        video.create_dataset("target_steps", data=np.arange(8))
    return checkpoint, gaussian, calibration


def _frame(value):
    return np.full((224, 224, 3), value, dtype=np.uint8)


def test_stateful_gaussian_progress_stride_and_ood_hold(tmp_path, monkeypatch):
    checkpoint, gaussian, calibration = _write_gaussian_assets(tmp_path)
    encoder = _RecordingEncoder()
    monkeypatch.setattr(
        BatchedGaussianProgressGatedProvider,
        "_build_encoder",
        lambda self: encoder,
    )
    provider = BatchedGaussianProgressGatedProvider(
        1,
        checkpoint_path=checkpoint,
        gaussian_model_h5_path=gaussian,
        calibration_h5_path=calibration,
        device="cpu",
        ood_p_value_threshold=0.3,
        posterior_temperature=1.0,
        frame_history_stride=2,
    )

    initial = provider.reset_all([_frame(0)]).copy()
    held = provider.advance_all([_frame(128)]).copy()
    updated = provider.advance_all([_frame(255)]).copy()
    provider.advance_all([_frame(64)])

    np.testing.assert_allclose(held, initial)
    assert updated[0] > initial[0] + 0.9
    np.testing.assert_allclose(encoder.contexts[0][0], [0.0, 0.0])
    np.testing.assert_allclose(
        encoder.contexts[1][0], [0.0, 128.0 / 255.0])
    np.testing.assert_allclose(encoder.contexts[2][0], [0.0, 1.0])
    np.testing.assert_allclose(
        encoder.contexts[3][0], [128.0 / 255.0, 64.0 / 255.0])


def test_replay_ten_step_pbrs_telescopes(tmp_path):
    gamma = 0.97
    potentials = np.linspace(0.1, 0.8, 11, dtype=np.float32)
    shaping = gamma * potentials[1:] - potentials[:-1]
    episode = {
        "observation": np.arange(11, dtype=np.uint8)[:, None],
        "action": np.arange(11, dtype=np.float32)[:, None],
        "reward": np.concatenate(
            [np.zeros(1, dtype=np.float32), shaping])[:, None],
        "discount": np.ones((11, 1), dtype=np.float32),
    }
    buffer = ReplayBuffer(
        tmp_path,
        max_size=100,
        num_workers=1,
        nstep=10,
        discount=gamma,
        fetch_every=1000,
        save_snapshot=True,
    )
    episode_key = Path("synthetic_episode")
    buffer._episode_fns = [episode_key]
    buffer._episodes = {episode_key: episode}

    _, _, reward, discount, next_observation = buffer._sample()

    expected = -potentials[0] + gamma ** 10 * potentials[10]
    assert reward.shape == (1,)
    assert discount.shape == (1,)
    assert reward[0] == pytest.approx(expected, abs=1.0e-6)
    assert discount[0] == pytest.approx(gamma ** 10, abs=1.0e-6)
    assert next_observation.tolist() == [10]


def test_shaped_reward_defaults_are_disabled_and_asset_free():
    root = Path(__file__).resolve().parents[1]
    with (root / "cfgs/config.yaml").open() as stream:
        base_cfg = yaml.safe_load(stream)
    with (root / "cfgs/task/metaworld.yaml").open() as stream:
        task_cfg = yaml.safe_load(stream)

    shaped = task_cfg["shaped_reward"]
    assert base_cfg["action_repeat"] == 2
    assert shaped["enabled"] is False
    assert shaped["pbrs_dense_scale"] == 1.0
    assert shaped["pbrs_gamma"] == 0.97
    assert shaped["reward_render_size"] == 224
    assert shaped["reward_crop_size"] is None
    assert shaped["reward_crop_offset"] is None
    for name in (
            "checkpoint_path",
            "gaussian_model_h5_path",
            "calibration_h5_path"):
        assert shaped[name] is None


def test_enabled_config_with_missing_threshold_fails_clearly():
    with pytest.raises(
            ValueError, match="shaped_reward.ood_p_value_threshold is required"):
        MetaWorldShapedRewardManager(
            {"ood_p_value_threshold": None}, device="cpu")
