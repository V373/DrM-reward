from pathlib import Path

import cv2
import imageio
import numpy as np
import wandb


def save_progress_composite_video(frames, progress, success, is_ood,
                                  output_path, fps=20):
    """Save raw frames beside animated progress and success curves."""
    frames = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    progress = np.asarray(progress, dtype=np.float32)
    success = np.asarray(success, dtype=np.float32)
    is_ood = np.asarray(is_ood, dtype=bool)
    if not frames:
        raise ValueError('progress video requires at least one frame')
    if (len(frames) != len(progress) or len(frames) != len(success)
            or len(frames) != len(is_ood)):
        raise ValueError(
            'progress video frames, progress, success, and is_ood must have '
            'equal length')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = frames[0].shape[0]
    panel_width = max(240, round(height * 1.5625))
    left, right = round(panel_width * 0.13), panel_width - 10
    ood_bounds = (max(2, round(height * 0.03)), max(3, round(height * 0.09)))
    top_bounds = (max(12, round(height * 0.16)), round(height * 0.45))
    bottom_bounds = (round(height * 0.62), round(height * 0.91))
    x_coords = np.linspace(left, right, len(frames)).round().astype(int)

    def y_coords(values, bounds):
        upper, lower = bounds
        clipped = np.clip(values, 0.0, 1.0)
        return np.rint(lower - clipped * (lower - upper)).astype(int)

    progress_y = y_coords(progress, top_bounds)
    success_y = y_coords(success, bottom_bounds)
    base_panel = np.full((height, panel_width, 3), 255, dtype=np.uint8)
    font_scale = max(0.25, height / 512.0)
    cv2.putText(base_panel, 'OOD', (left, max(9, ood_bounds[0] - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (20, 20, 20), 1, cv2.LINE_AA)
    for index, ood in enumerate(is_ood):
        start_x = int(x_coords[index])
        end_x = (int(x_coords[index + 1]) if index + 1 < len(frames)
                 else right)
        cv2.rectangle(
            base_panel,
            (start_x, ood_bounds[0]),
            (max(start_x, end_x), ood_bounds[1]),
            (214, 39, 40) if ood else (255, 255, 255),
            cv2.FILLED,
        )
    cv2.rectangle(base_panel, (left, ood_bounds[0]),
                  (right, ood_bounds[1]), (80, 80, 80), 1)
    for title, bounds in (
            ('Progress', top_bounds), ('Env Success', bottom_bounds)):
        upper, lower = bounds
        cv2.rectangle(base_panel, (left, upper), (right, lower),
                      (80, 80, 80), 1)
        for value in (0.0, 0.5, 1.0):
            y = round(lower - value * (lower - upper))
            cv2.line(base_panel, (left, y), (right, y),
                     (220, 220, 220), 1)
        cv2.putText(base_panel, title, (left, max(9, upper - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(base_panel, '0', (max(1, left - 14), lower + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                    (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(base_panel, '1', (max(1, left - 14), upper + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                    (60, 60, 60), 1, cv2.LINE_AA)

    progress_points = np.column_stack((x_coords, progress_y)).astype(np.int32)
    cv2.polylines(base_panel, [progress_points], False,
                  (31, 119, 180), 2, cv2.LINE_AA)
    success_points = [(int(x_coords[0]), int(success_y[0]))]
    for index in range(1, len(frames)):
        success_points.append((int(x_coords[index]), int(success_y[index - 1])))
        success_points.append((int(x_coords[index]), int(success_y[index])))
    cv2.polylines(base_panel, [np.asarray(success_points, dtype=np.int32)],
                  False, (148, 103, 189), 2, cv2.LINE_AA)

    def draw_cursor(panel, x, bounds):
        for y in range(bounds[0], bounds[1] + 1, 8):
            cv2.line(panel, (x, y), (x, min(y + 4, bounds[1])),
                     (0, 0, 0), 1)

    writer = None
    try:
        writer = imageio.get_writer(
            str(output_path), fps=fps, format='ffmpeg', codec='libx264',
            macro_block_size=1)
        for index, raw_frame in enumerate(frames):
            if raw_frame.ndim != 3 or raw_frame.shape[2] != 3:
                raise ValueError(
                    f'progress video frame must be HxWx3, got {raw_frame.shape}')
            if raw_frame.shape[0] != height:
                raise ValueError('all progress video frames need equal height')
            panel = base_panel.copy()
            for bounds in (ood_bounds, top_bounds, bottom_bounds):
                draw_cursor(panel, int(x_coords[index]), bounds)
            composite = np.concatenate([raw_frame, panel], axis=1)
            composite = composite[
                :composite.shape[0] - composite.shape[0] % 2,
                :composite.shape[1] - composite.shape[1] % 2]
            writer.append_data(composite)
    finally:
        if writer is not None:
            writer.close()
    return output_path


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.use_wandb = use_wandb
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def init_obs(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record_obs(obs)

    def record_obs(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.get_pixels_with_width_height(self.render_size, 
                                                         self.render_size)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            if "adroit" in str(self.save_dir):
                import numpy as np
                self.frames = np.array(self.frames, dtype=np.uint8).transpose(0, 2, 3, 1)
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            if self.use_wandb and wandb.run is not None:
                try:
                    # MetaWorld default timing in train_mw: native env.step() is
                    # 80 Hz and action_repeat=2, while this recorder stores one
                    # frame per outer step. The recorded states are therefore
                    # 40 Hz, but self.fps is 20 by default, so eval/video plays
                    # at 2x slower than simulated time (unless either setting
                    # is overridden).
                    wandb.log({
                        'eval/video': wandb.Video(str(path),
                                                  fps=self.fps,
                                                  format='mp4')
                    }, commit=False)
                except Exception as exc:
                    print(f'[wandb] failed to upload eval video {path}: {exc}')

    def save_progress(self, file_name, reward_frames, progress, success, is_ood):
        if not self.enabled:
            return None
        file_path = Path(file_name)
        path = self.save_dir / f'{file_path.stem}_progress.mp4'
        save_progress_composite_video(
            reward_frames[1:], progress, success, is_ood, path, fps=self.fps)
        if self.use_wandb and wandb.run is not None:
            try:
                wandb.log({
                    'eval/progress_video': wandb.Video(
                        str(path), fps=self.fps, format='mp4')
                }, commit=False)
            except Exception as exc:
                print(
                    f'[wandb] failed to upload eval progress video {path}: {exc}')
        return path


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
