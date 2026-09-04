"""MetaWorld adapter for image-based dense and PBRS rewards."""

from pathlib import Path

import numpy as np

from .gaussian_progress import BatchedGaussianProgressGatedProvider


_DRM_ROOT = Path(__file__).resolve().parents[1]


def _required_path(value, name):
    if value is None or str(value).strip().lower() in ("", "none"):
        raise ValueError(f"shaped_reward.{name} is required")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _DRM_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"shaped_reward.{name} not found: {path}")
    return str(path)


class MetaWorldShapedRewardManager:
    def __init__(self, cfg, device):
        self.reward_type = str(cfg.get("type", "pbrs"))
        if self.reward_type not in ("dense", "pbrs"):
            raise ValueError("shaped_reward.type must be dense or pbrs")

        self.sparse_scale = float(cfg.get("sparse_scale", 1.0))
        self.shaping_scale = float(cfg.get("pbrs_dense_scale", 1.0))
        self.pbrs_gamma = float(cfg.get("pbrs_gamma", 0.97))
        if not all(np.isfinite(value) for value in (
                self.sparse_scale, self.shaping_scale, self.pbrs_gamma)):
            raise ValueError("shaped reward scales and gamma must be finite")
        self.bias_enabled = bool(
            cfg.get("pbrs_progress_bias_enabled", False))
        bias_b = float(cfg.get("pbrs_progress_bias_b", 0.0))
        if not np.isfinite(bias_b):
            raise ValueError("pbrs_progress_bias_b must be finite")
        if self.bias_enabled:
            if abs(self.pbrs_gamma - 1.0) <= 1.0e-12:
                raise ValueError("PBRS progress bias requires pbrs_gamma != 1")
            self.progress_bias = bias_b / (self.pbrs_gamma - 1.0)
        else:
            self.progress_bias = 0.0

        self.exp_enabled = bool(
            cfg.get("pbrs_progress_exp_enabled", False))
        self.exp_base = float(cfg.get("pbrs_progress_exp_base", 32.0))
        if self.exp_enabled and (
                not np.isfinite(self.exp_base) or self.exp_base <= 0.0):
            raise ValueError(
                "pbrs_progress_exp_base must be finite and positive")

        threshold = cfg.get("ood_p_value_threshold", None)
        if threshold is None:
            raise ValueError("shaped_reward.ood_p_value_threshold is required")
        self.provider = BatchedGaussianProgressGatedProvider(
            1,
            checkpoint_path=_required_path(
                cfg.get("checkpoint_path", None), "checkpoint_path"),
            gaussian_model_h5_path=_required_path(
                cfg.get("gaussian_model_h5_path", None),
                "gaussian_model_h5_path"),
            calibration_h5_path=_required_path(
                cfg.get("calibration_h5_path", None),
                "calibration_h5_path"),
            device=device,
            ood_p_value_threshold=float(threshold),
            posterior_temperature=float(
                cfg.get("posterior_temperature", 1.0e4)),
            frame_history_stride=int(cfg.get("frame_history_stride", 4)),
            enable_ood_filter=cfg.get("enable_ood_filter", False),
            ood_filter_max_gap=cfg.get("ood_filter_max_gap", None),
            ood_filter_min_ood_run=cfg.get(
                "ood_filter_min_ood_run", None),
        )
        self.enable_ood_filter = self.provider.enable_ood_filter

    def reset(self, frame):
        progress = float(self.provider.reset_all([frame])[0])
        if not np.isfinite(progress):
            raise ValueError("Initial progress is NaN or Inf")
        return progress

    def _compute_reward(self, sparse_reward, progress, inferred_next, done):
        sparse_reward = float(sparse_reward)
        progress = float(progress)
        inferred_next = float(inferred_next)
        if not all(np.isfinite(value) for value in (
                sparse_reward, progress, inferred_next)):
            raise ValueError("Sparse reward and progress values must be finite")

        if self.reward_type == "dense":
            progress_next = 0.0
            raw_shaping = progress
        else:
            progress_next = 0.0 if done else inferred_next
            current_potential = progress + self.progress_bias
            next_potential = progress_next + self.progress_bias
            if self.exp_enabled:
                current_potential = self.exp_base ** current_potential
                next_potential = self.exp_base ** next_potential
            if done:
                next_potential = 0.0
            raw_shaping = (
                self.pbrs_gamma * next_potential - current_potential)

        shaping_reward = self.shaping_scale * raw_shaping
        shaped_reward = (
            self.sparse_scale * sparse_reward + shaping_reward)
        if not np.isfinite(shaped_reward):
            raise ValueError("Shaped reward is NaN or Inf")

        metrics = {
            "shaped_reward/raw_sparse": float(sparse_reward),
            "shaped_reward/progress": progress,
            "shaped_reward/progress_next": progress_next,
            "shaped_reward/shaping": float(shaping_reward),
            "shaped_reward/total": float(shaped_reward),
        }
        return float(shaped_reward), metrics

    def step(self, sparse_reward, next_frame, done):
        if self.enable_ood_filter:
            raise RuntimeError(
                "step() is unavailable when episode finalization is enabled")
        if self.provider.progress_current is None:
            raise RuntimeError("Shaped reward manager must be reset before step")

        progress = float(self.provider.progress_current[0])
        inferred_next = float(self.provider.advance_all(
            [next_frame], reset_mask=np.array([False]))[0])
        return self._compute_reward(
            sparse_reward, progress, inferred_next, done)

    def advance(self, next_frame):
        if not self.enable_ood_filter:
            raise RuntimeError(
                "advance() requires episode finalization to be enabled")
        if self.provider.progress_current is None:
            raise RuntimeError("Shaped reward manager must be reset before advance")
        progress = float(self.provider.advance_all(
            [next_frame], reset_mask=np.array([False]))[0])
        if not np.isfinite(progress):
            raise ValueError("Progress is NaN or Inf")
        return progress

    def finalize_episode(self, sparse_rewards, dones):
        if not self.enable_ood_filter:
            raise RuntimeError(
                "finalize_episode() requires enable_ood_filter=true")
        sparse_rewards = np.asarray(sparse_rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=np.bool_)
        if sparse_rewards.ndim != 1 or dones.ndim != 1:
            raise ValueError("sparse_rewards and dones must be one-dimensional")
        if sparse_rewards.size == 0 or sparse_rewards.shape != dones.shape:
            raise ValueError(
                "sparse_rewards and dones must have the same non-empty shape")
        if not np.isfinite(sparse_rewards).all():
            raise ValueError("Sparse rewards contain NaN or Inf")

        progress_final = np.asarray(
            self.provider.finalize_episode(), dtype=np.float64)
        if progress_final.ndim != 1 or progress_final.size != sparse_rewards.size + 1:
            raise ValueError(
                "Finalized progress length must equal transition count plus one")
        if not np.isfinite(progress_final).all():
            raise ValueError("Finalized progress contains NaN or Inf")

        rewards = []
        metrics = []
        for index, sparse_reward in enumerate(sparse_rewards):
            reward, step_metrics = self._compute_reward(
                sparse_reward=sparse_reward,
                progress=progress_final[index],
                inferred_next=progress_final[index + 1],
                done=bool(dones[index]))
            rewards.append(reward)
            metrics.append(step_metrics)
        return rewards, metrics

    def infer_episode_progress(self, frames):
        return self.provider.infer_episode_trace(frames)


class MetaWorldShapedRewardWrapper:
    """Replace a single MetaWorld training env's sparse reward."""

    def __init__(self, env, cfg, device, manager=None):
        self._env = env
        self._camera = str(cfg.get("reward_camera", "corner2"))
        self._manager = manager or MetaWorldShapedRewardManager(cfg, device)
        self.last_reward_metrics = {}

    @property
    def requires_episode_finalize(self):
        return bool(getattr(self._manager, "enable_ood_filter", False))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        time_step = self._env.reset()
        self._manager.reset(self._env.get_reward_frame(self._camera))
        self.last_reward_metrics = {}
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        next_frame = self._env.get_reward_frame(self._camera)
        if self.requires_episode_finalize:
            self._manager.advance(next_frame)
            self.last_reward_metrics = {}
            return time_step
        reward, self.last_reward_metrics = self._manager.step(
            sparse_reward=time_step.reward,
            next_frame=next_frame,
            done=time_step.last())
        return time_step._replace(reward=reward)

    def finalize_episode(self, episode_pending):
        if not self.requires_episode_finalize:
            raise RuntimeError(
                "finalize_episode() requires enable_ood_filter=true")
        episode_pending = list(episode_pending)
        if len(episode_pending) < 2:
            raise ValueError("Pending episode must contain FIRST and LAST timesteps")
        if not episode_pending[0].first():
            raise ValueError("Pending episode must start with a FIRST timestep")
        if any(time_step.first() for time_step in episode_pending[1:]):
            raise ValueError("Pending episode contains an unexpected FIRST timestep")
        if not episode_pending[-1].last():
            raise ValueError("Pending episode must end with a LAST timestep")
        if any(time_step.last() for time_step in episode_pending[1:-1]):
            raise ValueError("Pending episode contains an early LAST timestep")

        transitions = episode_pending[1:]
        rewards, metrics = self._manager.finalize_episode(
            sparse_rewards=[time_step.reward for time_step in transitions],
            dones=[time_step.last() for time_step in transitions])
        if len(rewards) != len(transitions) or len(metrics) != len(transitions):
            raise RuntimeError(
                "Finalized rewards and metrics must match transition count")
        finalized_episode = [episode_pending[0]]
        finalized_episode.extend(
            time_step._replace(reward=reward)
            for time_step, reward in zip(transitions, rewards))
        self.last_reward_metrics = metrics[-1]
        return finalized_episode, metrics

    def infer_episode_progress(self, frames):
        return self._manager.infer_episode_progress(frames)
