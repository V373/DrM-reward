"""Stateful Gaussian progress inference, localized from FineProg."""

from pathlib import Path
import warnings

import h5py
import numpy as np
from scipy.linalg import solve_triangular
import torch

from .encoder import TCCEncoder


_MODEL_DATASETS = (
    "bin_progress_values",
    "bin_means",
    "bin_independent_covariances",
    "shared_covariance",
    "bin_final_covariances",
    "bin_log_determinants",
    "bin_counts",
)


def _validate_normalization(value, context):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value not in ("none", "l2"):
        raise ValueError(
            f"{context}: embedding_normalization must be 'none' or 'l2', "
            f"got {value!r}")
    return value


def _read_normalization(h5_file, path):
    if "embedding_normalization" not in h5_file.attrs:
        warnings.warn(
            f"{path}: missing embedding_normalization; treating it as 'none'",
            UserWarning,
            stacklevel=2)
        return "none"
    return _validate_normalization(
        h5_file.attrs["embedding_normalization"], str(path))


def _validate_embeddings(embeddings, normalization, context):
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError(f"{context}: embeddings must be finite [T, D] values")
    if normalization == "l2":
        norms = np.linalg.norm(embeddings.astype(np.float64, copy=False), axis=1)
        if not np.all(np.isclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-5)):
            raise ValueError(
                f"{context}: embedding_normalization='l2' requires unit rows")


def _positive_int_attr(attrs, name, required=True):
    if name not in attrs:
        if required:
            raise ValueError(f"Gaussian model is missing attribute {name!r}")
        return None
    value = attrs[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"Gaussian model attribute {name!r} must be a positive integer")
    return int(value)


def _read_gaussian_model(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian model H5 not found: {path}")

    with h5py.File(path, "r") as h5_file:
        normalization = _read_normalization(h5_file, path)
        enable_pca_value = h5_file.attrs.get("enable_pca", False)
        if not isinstance(enable_pca_value, (bool, np.bool_)):
            raise ValueError("Gaussian model enable_pca must be boolean")
        enable_pca = bool(enable_pca_value)

        if "model" not in h5_file or not isinstance(h5_file["model"], h5py.Group):
            raise ValueError("Gaussian model H5 must contain /model")
        model_group = h5_file["model"]
        missing = [name for name in _MODEL_DATASETS if name not in model_group]
        if missing:
            raise ValueError(f"Gaussian model H5 is missing datasets: {missing}")

        progress_values = np.asarray(
            model_group["bin_progress_values"][:], dtype=np.float64)
        means = np.asarray(model_group["bin_means"][:], dtype=np.float64)
        covariances = np.asarray(
            model_group["bin_final_covariances"][:], dtype=np.float64)
        log_determinants = np.asarray(
            model_group["bin_log_determinants"][:], dtype=np.float64)

        if enable_pca:
            missing_pca = [
                name for name in ("pca_mean", "pca_components")
                if name not in model_group
            ]
            if missing_pca:
                raise ValueError(
                    f"PCA Gaussian model is missing datasets: {missing_pca}")
            pca_mean = np.asarray(model_group["pca_mean"][:], dtype=np.float64)
            pca_components = np.asarray(
                model_group["pca_components"][:], dtype=np.float64)
            embedding_dim = _positive_int_attr(h5_file.attrs, "embedding_dim")
            input_embedding_dim = _positive_int_attr(
                h5_file.attrs, "input_embedding_dim")
        else:
            if "pca_mean" in model_group or "pca_components" in model_group:
                raise ValueError(
                    "enable_pca=false is inconsistent with saved PCA datasets")
            pca_mean = None
            pca_components = None
            embedding_dim = _positive_int_attr(
                h5_file.attrs, "embedding_dim", required=False)
            input_embedding_dim = _positive_int_attr(
                h5_file.attrs, "input_embedding_dim", required=False)

    if progress_values.ndim != 1 or progress_values.size < 2:
        raise ValueError("bin_progress_values must have shape [K] with K >= 2")
    num_bins = progress_values.size
    if means.ndim != 2 or means.shape[0] != num_bins or means.shape[1] < 1:
        raise ValueError("bin_means must have shape [K, D]")
    inferred_dim = means.shape[1]
    if embedding_dim is not None and embedding_dim != inferred_dim:
        raise ValueError("Gaussian embedding_dim does not match bin_means")
    embedding_dim = inferred_dim

    if enable_pca:
        if input_embedding_dim <= embedding_dim:
            raise ValueError("PCA input_embedding_dim must exceed embedding_dim")
        if pca_mean.shape != (input_embedding_dim,):
            raise ValueError("pca_mean has an invalid shape")
        if pca_components.shape != (embedding_dim, input_embedding_dim):
            raise ValueError("pca_components has an invalid shape")
    else:
        if input_embedding_dim is not None and input_embedding_dim != embedding_dim:
            raise ValueError("input_embedding_dim must equal embedding_dim without PCA")
        input_embedding_dim = embedding_dim

    if covariances.shape != (num_bins, embedding_dim, embedding_dim):
        raise ValueError("bin_final_covariances must have shape [K, D, D]")
    if log_determinants.shape != (num_bins,):
        raise ValueError("bin_log_determinants must have shape [K]")
    arrays = [progress_values, means, covariances, log_determinants]
    if enable_pca:
        arrays.extend((pca_mean, pca_components))
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Gaussian model contains NaN or Inf")
    if np.any(progress_values < 0.0) or np.any(progress_values > 1.0):
        raise ValueError("bin_progress_values must lie in [0, 1]")

    cholesky = np.empty_like(covariances)
    for index, covariance in enumerate(covariances):
        if not np.allclose(covariance, covariance.T,
                           rtol=1.0e-8, atol=1.0e-10):
            raise ValueError(f"covariance {index} is not symmetric")
        try:
            cholesky[index] = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"covariance {index} is not positive definite") from error

    return {
        "bin_progress_values": progress_values,
        "bin_means": means,
        "bin_log_determinants": log_determinants,
        "cholesky_factors": cholesky,
        "num_bins": num_bins,
        "enable_pca": enable_pca,
        "input_embedding_dim": input_embedding_dim,
        "embedding_dim": embedding_dim,
        "embedding_normalization": normalization,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }


def _apply_pca(embeddings, model):
    if not model["enable_pca"]:
        return embeddings
    projected = (
        embeddings - model["pca_mean"][None]
    ) @ model["pca_components"].T
    if projected.shape[1] != model["embedding_dim"] or not np.isfinite(
            projected).all():
        raise ValueError("PCA-projected embeddings are invalid")
    return projected


def _squared_mahalanobis(embeddings, means, cholesky):
    distances = np.empty((len(embeddings), len(means)), dtype=np.float64)
    for index in range(len(means)):
        deltas = embeddings - means[index]
        whitened = solve_triangular(
            cholesky[index], deltas.T, lower=True, check_finite=False)
        distances[:, index] = np.einsum(
            "dt,dt->t", whitened, whitened, optimize=True)
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise ValueError("Mahalanobis distances are invalid")
    return distances


def _temporal_progress(num_embeddings, target_steps):
    if num_embeddings < 2:
        raise ValueError("calibration trajectories need at least two embeddings")
    steps = np.asarray(target_steps, dtype=np.float64)
    if steps.shape != (num_embeddings,) or not np.isfinite(steps).all():
        raise ValueError("calibration target_steps are invalid")
    if steps[-1] == steps[0]:
        return np.arange(num_embeddings, dtype=np.float64) / (num_embeddings - 1)
    progress = (steps - steps[0]) / (steps[-1] - steps[0])
    if not np.isfinite(progress).all() or progress.min() < -1.0e-12 \
            or progress.max() > 1.0 + 1.0e-12:
        raise ValueError("calibration temporal progress must lie in [0, 1]")
    return np.clip(progress, 0.0, 1.0)


def _read_calibration_bins(path, model):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration H5 not found: {path}")

    per_bin = [[] for _ in range(model["num_bins"])]
    with h5py.File(path, "r") as h5_file:
        normalization = _read_normalization(h5_file, path)
        if normalization != model["embedding_normalization"]:
            raise ValueError(
                "embedding_normalization differs between Gaussian and calibration H5")
        if "videos" not in h5_file or not isinstance(h5_file["videos"], h5py.Group):
            raise ValueError("Calibration H5 must contain /videos")
        videos = h5_file["videos"]
        if len(videos) == 0:
            raise ValueError("Calibration H5 /videos is empty")

        for video_id in sorted(videos.keys()):
            video = videos[video_id]
            if "embeddings" not in video or "target_steps" not in video:
                raise ValueError(
                    f"Calibration video {video_id} needs embeddings and target_steps")
            embeddings = np.asarray(video["embeddings"][:], dtype=np.float64)
            if embeddings.ndim != 2 or embeddings.shape[1] != model[
                    "input_embedding_dim"]:
                raise ValueError(
                    f"Calibration video {video_id} has an invalid embedding shape")
            _validate_embeddings(
                embeddings, normalization,
                f"{path}:/videos/{video_id}/embeddings")
            progress = _temporal_progress(
                len(embeddings), np.asarray(video["target_steps"][:]))
            projected = _apply_pca(embeddings, model)
            distances = _squared_mahalanobis(
                projected, model["bin_means"], model["cholesky_factors"])
            indices = np.minimum(
                np.floor(model["num_bins"] * progress).astype(np.int64),
                model["num_bins"] - 1)
            for index in range(model["num_bins"]):
                selected = distances[indices == index, index]
                if selected.size:
                    per_bin[index].append(selected)

    empty = [index for index, parts in enumerate(per_bin) if not parts]
    if empty:
        raise ValueError(f"Calibration temporal bins have no samples: {empty}")
    return [np.concatenate(parts) for parts in per_bin]


def _filter_is_ood_short_id_gaps(is_ood, max_gap, min_ood_run):
    """Fill short internal ID gaps bounded by sufficiently long OOD runs."""
    raw_is_ood = np.asarray(is_ood)
    if raw_is_ood.ndim != 1 or raw_is_ood.dtype != np.dtype("bool"):
        raise ValueError("is_ood must be a one-dimensional boolean array")
    for name, value in (("max_gap", max_gap), ("min_ood_run", min_ood_run)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")

    max_gap = int(max_gap)
    min_ood_run = int(min_ood_run)
    filtered_is_ood = raw_is_ood.copy()
    num_steps = int(raw_is_ood.size)
    index = 0
    left_ood_run = 0

    while index < num_steps:
        if raw_is_ood[index]:
            run_start = index
            while index < num_steps and raw_is_ood[index]:
                index += 1
            left_ood_run = index - run_start
            continue

        gap_start = index
        while index < num_steps and not raw_is_ood[index]:
            index += 1
        gap_end = index

        if gap_start == 0 or gap_end == num_steps:
            left_ood_run = 0
            continue

        right_run_start = index
        while index < num_steps and raw_is_ood[index]:
            index += 1
        right_ood_run = index - right_run_start

        if (gap_end - gap_start <= max_gap
                and left_ood_run >= min_ood_run
                and right_ood_run >= min_ood_run):
            filtered_is_ood[gap_start:gap_end] = True
        left_ood_run = right_ood_run

    return filtered_is_ood


def _compute_progress_gated(progress_mean, is_ood):
    """Hold the last in-distribution progress across OOD frames."""
    progress_gated = np.empty_like(progress_mean)
    last_progress = 0.0
    for frame_index in range(progress_mean.shape[0]):
        if not is_ood[frame_index]:
            last_progress = progress_mean[frame_index]
        progress_gated[frame_index] = last_progress
    return progress_gated


class BatchedGaussianProgressGatedProvider:
    """Infer OOD-gated progress for one or more online environments."""

    image_height = 224
    image_width = 224

    def __init__(
            self,
            n_envs,
            checkpoint_path,
            gaussian_model_h5_path,
            calibration_h5_path,
            device,
            *,
            ood_p_value_threshold,
            posterior_temperature=1.0e4,
            frame_history_stride=4,
            enable_ood_filter=False,
            ood_filter_max_gap=None,
            ood_filter_min_ood_run=None):
        self.n_envs = int(n_envs)
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.device = torch.device(device)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.posterior_temperature = float(posterior_temperature)
        self.ood_p_value_threshold = float(ood_p_value_threshold)
        self.frame_history_stride = int(frame_history_stride)
        if self.frame_history_stride < 1:
            raise ValueError("frame_history_stride must be >= 1")
        if not isinstance(enable_ood_filter, (bool, np.bool_)):
            raise ValueError("enable_ood_filter must be boolean")
        self.enable_ood_filter = bool(enable_ood_filter)
        if self.enable_ood_filter:
            filter_values = {
                "ood_filter_max_gap": ood_filter_max_gap,
                "ood_filter_min_ood_run": ood_filter_min_ood_run,
            }
            for name, value in filter_values.items():
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                        value, (int, np.integer)) or int(value) < 1:
                    raise ValueError(
                        f"{name} must be a positive integer when "
                        "enable_ood_filter=true")
            self.ood_filter_max_gap = int(filter_values["ood_filter_max_gap"])
            self.ood_filter_min_ood_run = int(
                filter_values["ood_filter_min_ood_run"])
        else:
            self.ood_filter_max_gap = None
            self.ood_filter_min_ood_run = None
        if not np.isfinite(self.posterior_temperature) \
                or self.posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be finite and > 0")
        if not 0.0 < self.ood_p_value_threshold < 1.0:
            raise ValueError("ood_p_value_threshold must be in (0, 1)")

        self.history_len = self.frame_history_stride + 1
        self.gaussian_model = _read_gaussian_model(gaussian_model_h5_path)
        self.calibration_bins = _read_calibration_bins(
            calibration_h5_path, self.gaussian_model)
        self.encoder = self._build_encoder()
        self._build_model_tensors()

        self._frame_ring = torch.zeros(
            (self.n_envs, self.history_len, 3,
             self.image_height, self.image_width),
            dtype=torch.uint8,
            device=self.device)
        self._staging = torch.empty(
            (self.n_envs, self.image_height, self.image_width, 3),
            dtype=torch.uint8,
            pin_memory=self.device.type == "cuda")
        self._staging_numpy = self._staging.numpy()
        self._env_indices = torch.arange(self.n_envs, device=self.device)
        self._head = torch.zeros(
            self.n_envs, dtype=torch.long, device=self.device)
        self._since_reset = torch.zeros(
            self.n_envs, dtype=torch.long, device=self.device)
        self._last_in_distribution = torch.zeros(
            self.n_envs, dtype=torch.float64, device=self.device)
        self._last_is_ood = torch.zeros(
            self.n_envs, dtype=torch.bool, device=self.device)
        self._episode_progress_raw_by_env = [
            [] for _ in range(self.n_envs)]
        self._episode_is_ood_raw_by_env = [
            [] for _ in range(self.n_envs)]
        self.progress_current = None

    def _build_encoder(self):
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Encoder checkpoint needs a model_state_dict mapping")
        if "embedding_normalization" not in checkpoint:
            raise ValueError("Encoder checkpoint needs embedding_normalization metadata")
        checkpoint_normalization = _validate_normalization(
            checkpoint["embedding_normalization"], self.checkpoint_path)
        model_normalization = self.gaussian_model["embedding_normalization"]
        if checkpoint_normalization != model_normalization:
            raise ValueError(
                "embedding_normalization differs between checkpoint and Gaussian H5")

        encoder = TCCEncoder(
            embedding_dim=self.gaussian_model["input_embedding_dim"],
            embedding_normalization=model_normalization).to(self.device)
        encoder.load_state_dict(checkpoint["model_state_dict"])
        encoder.eval()
        return encoder

    def _build_model_tensors(self):
        def as_tensor(array):
            return torch.as_tensor(
                np.asarray(array, dtype=np.float64), device=self.device)

        model = self.gaussian_model
        self._bin_means = as_tensor(model["bin_means"])
        self._bin_log_determinants = as_tensor(model["bin_log_determinants"])
        self._bin_progress_values = as_tensor(model["bin_progress_values"])
        self._whitening = as_tensor(
            np.linalg.inv(np.asarray(model["cholesky_factors"], dtype=np.float64)))
        self._gaussian_constant = model["embedding_dim"] * np.log(2.0 * np.pi)
        if model["enable_pca"]:
            self._pca_mean = as_tensor(model["pca_mean"])
            self._pca_components_t = as_tensor(
                model["pca_components"]).T.contiguous()
        else:
            self._pca_mean = None
            self._pca_components_t = None

        counts = np.asarray([len(values) for values in self.calibration_bins])
        padded = np.full((len(counts), int(counts.max())),
                         np.inf, dtype=np.float64)
        for index, values in enumerate(self.calibration_bins):
            padded[index, :len(values)] = np.sort(values)
        self._calibration_sorted = as_tensor(padded)
        self._calibration_counts = torch.as_tensor(
            counts, dtype=torch.long, device=self.device)

    def _write_frames(self, frames):
        if len(frames) != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} reward frames, got {len(frames)}")
        expected_shape = (self.image_height, self.image_width, 3)
        for index, frame in enumerate(frames):
            frame = np.asarray(frame)
            if frame.shape != expected_shape or frame.dtype != np.uint8:
                raise ValueError(
                    f"Reward frame {index} must be uint8 {expected_shape}, "
                    f"got {frame.dtype} {frame.shape}")
            self._staging_numpy[index] = frame
        return self._staging.to(
            self.device, non_blocking=True).permute(0, 3, 1, 2)

    def _infer_batch(self):
        old_index = (self._head - self._since_reset) % self.history_len
        context = torch.stack((
            self._frame_ring[self._env_indices, old_index],
            self._frame_ring[self._env_indices, self._head],
        ), dim=1)
        context = context.to(torch.float32).div_(255.0).unsqueeze(1)
        embeddings = self.encoder(context)
        expected_shape = (
            self.n_envs, 1, self.gaussian_model["input_embedding_dim"])
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Encoder output must be a torch.Tensor")
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(
                f"Encoder output must have shape {expected_shape}, "
                f"got {tuple(embeddings.shape)}")
        if not torch.isfinite(embeddings).all():
            raise ValueError("Encoder output contains NaN or Inf")

        query = embeddings[:, 0].to(torch.float64)
        if self._pca_components_t is not None:
            query = (query - self._pca_mean) @ self._pca_components_t
        deltas = query.unsqueeze(1) - self._bin_means.unsqueeze(0)
        whitened = torch.einsum("kij,nkj->nki", self._whitening, deltas)
        squared_mahalanobis = (whitened * whitened).sum(dim=-1)
        log_likelihood = -0.5 * (
            squared_mahalanobis
            + self._bin_log_determinants
            + self._gaussian_constant)
        posterior = torch.softmax(
            log_likelihood / self.posterior_temperature, dim=1)
        progress = (posterior @ self._bin_progress_values).clamp(0.0, 1.0)

        num_greater = self._calibration_counts.unsqueeze(1) - torch.searchsorted(
            self._calibration_sorted,
            squared_mahalanobis.T.contiguous(),
            right=True)
        p_values = (1.0 + num_greater.T.to(torch.float64)) / (
            1.0 + self._calibration_counts.to(torch.float64))
        is_ood = (p_values < self.ood_p_value_threshold).all(dim=1)
        self._last_in_distribution = torch.where(
            is_ood, self._last_in_distribution, progress)
        self._last_is_ood = is_ood
        return self._last_in_distribution, progress, is_ood

    def _finalize_trace(self, progress_raw, is_ood_raw):
        if not self.enable_ood_filter:
            raise RuntimeError(
                "finalize_episode() requires enable_ood_filter=true")
        if not progress_raw:
            raise RuntimeError(
                "Episode trace is empty; reset the provider before finalizing")
        if len(progress_raw) != len(is_ood_raw):
            raise RuntimeError(
                "Episode progress and OOD traces have different lengths")

        progress_raw = np.asarray(progress_raw, dtype=np.float64)
        is_ood_raw = np.asarray(is_ood_raw, dtype=np.bool_)
        is_ood_final = _filter_is_ood_short_id_gaps(
            is_ood_raw,
            max_gap=self.ood_filter_max_gap,
            min_ood_run=self.ood_filter_min_ood_run)
        progress_final = _compute_progress_gated(progress_raw, is_ood_final)
        return progress_final, is_ood_final

    def advance_all(self, frames, reset_mask=None):
        if reset_mask is None:
            if self.progress_current is None:
                raise RuntimeError("Provider must be reset before advance_all")
            reset = np.zeros(self.n_envs, dtype=bool)
        else:
            reset = np.asarray(reset_mask, dtype=bool).reshape(-1)
            if reset.shape != (self.n_envs,):
                raise ValueError(
                    f"reset_mask must have shape ({self.n_envs},), got {reset.shape}")
            if self.progress_current is None and not reset.all():
                raise RuntimeError("Provider must be reset before advance_all")

        with torch.inference_mode():
            is_reset = torch.as_tensor(reset, device=self.device)
            if self.enable_ood_filter:
                for env_index in np.flatnonzero(reset):
                    self._episode_progress_raw_by_env[int(env_index)].clear()
                    self._episode_is_ood_raw_by_env[int(env_index)].clear()
            self._head = (self._head + 1) % self.history_len
            self._since_reset = torch.where(
                is_reset,
                torch.zeros_like(self._since_reset),
                (self._since_reset + 1).clamp(max=self.history_len - 1))
            self._last_in_distribution = torch.where(
                is_reset,
                torch.zeros_like(self._last_in_distribution),
                self._last_in_distribution)
            self._frame_ring[self._env_indices, self._head] = self._write_frames(frames)
            inferred_progress, progress_raw, is_ood_raw = self._infer_batch()
            if self.enable_ood_filter:
                progress_raw = progress_raw.cpu().numpy()
                is_ood_raw = is_ood_raw.cpu().numpy()
                for env_index in range(self.n_envs):
                    self._episode_progress_raw_by_env[env_index].append(
                        float(progress_raw[env_index]))
                    self._episode_is_ood_raw_by_env[env_index].append(
                        bool(is_ood_raw[env_index]))
            self.progress_current = inferred_progress.cpu().numpy()
        return self.progress_current

    def reset_all(self, frames):
        self.progress_current = None
        return self.advance_all(
            frames, reset_mask=np.ones(self.n_envs, dtype=bool))

    def finalize_episode(self, env_index=None):
        """Finalize one environment's complete progress trajectory."""
        if env_index is None:
            if self.n_envs != 1:
                raise ValueError(
                    "env_index is required when finalizing a multi-env provider")
            env_index = 0
        if isinstance(env_index, (bool, np.bool_)) or not isinstance(
                env_index, (int, np.integer)):
            raise ValueError("env_index must be an integer")
        env_index = int(env_index)
        if env_index < 0 or env_index >= self.n_envs:
            raise ValueError(
                f"env_index must be in [0, {self.n_envs}), got {env_index}")
        progress_final, _ = self._finalize_trace(
            self._episode_progress_raw_by_env[env_index],
            self._episode_is_ood_raw_by_env[env_index])
        return progress_final

    def infer_episode_trace(self, frames):
        """Infer transition progress without changing the online state."""
        if self.n_envs != 1:
            raise ValueError("Episode trace inference requires n_envs=1")
        frames = list(frames)
        if not frames:
            raise ValueError("Episode trace inference requires an initial frame")

        saved_frame_ring = self._frame_ring.clone()
        saved_head = self._head.clone()
        saved_since_reset = self._since_reset.clone()
        saved_last_in_distribution = self._last_in_distribution.clone()
        saved_last_is_ood = self._last_is_ood.clone()
        saved_progress_raw = [
            list(trace) for trace in self._episode_progress_raw_by_env]
        saved_is_ood_raw = [
            list(trace) for trace in self._episode_is_ood_raw_by_env]
        saved_progress = (
            None if self.progress_current is None
            else np.array(self.progress_current, copy=True))
        try:
            current = self.reset_all([frames[0]])
            trace = []
            ood_trace = []
            for frame in frames[1:]:
                trace.append(float(current[0]))
                ood_trace.append(bool(self._last_is_ood[0].item()))
                current = self.advance_all([frame])
            if self.enable_ood_filter:
                progress_final, is_ood_final = self._finalize_trace(
                    self._episode_progress_raw_by_env[0],
                    self._episode_is_ood_raw_by_env[0])
                return (
                    np.asarray(progress_final[:-1], dtype=np.float32),
                    np.asarray(is_ood_final[:-1], dtype=bool),
                )
            return (
                np.asarray(trace, dtype=np.float32),
                np.asarray(ood_trace, dtype=bool),
            )
        finally:
            with torch.inference_mode():
                self._frame_ring.copy_(saved_frame_ring)
                self._head.copy_(saved_head)
                self._since_reset.copy_(saved_since_reset)
                self._last_in_distribution.copy_(
                    saved_last_in_distribution)
                self._last_is_ood.copy_(saved_last_is_ood)
            self._episode_progress_raw_by_env = saved_progress_raw
            self._episode_is_ood_raw_by_env = saved_is_ood_raw
            self.progress_current = saved_progress
