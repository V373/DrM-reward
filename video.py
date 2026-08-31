import cv2
import imageio
import wandb


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
