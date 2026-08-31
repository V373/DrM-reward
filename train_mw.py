import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import metaworld_env as mw

from logger import Logger
from parallel_env import ParallelMetaWorld
from replay_buffer import ReplayBufferStorage, make_replay_loader
from shaped_reward import MetaWorldShapedRewardWrapper
from video import TrainVideoRecorder, VideoRecorder
import wandb
import math
import re
import time
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.task_name, str(cfg.seed)])
            exp_name = f'{cfg.wandb_run_name_prefix}{exp_name}'
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            wandb.init(project="DrM",
                       group=group_name,
                       name=exp_name,
                       config=cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._discount_alpha = cfg.discount_alpha
        self._discount_alpha_temp = cfg.discount_alpha_temp
        self._discount_beta = cfg.discount_beta
        self._discount_beta_temp = cfg.discount_beta_temp
        self._nstep = cfg.nstep
        self._nstep_alpha = cfg.nstep_alpha
        self._nstep_alpha_temp = cfg.nstep_alpha_temp
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        shaped_cfg = self.cfg.get('shaped_reward', None)
        self.shaped_reward_enabled = bool(
            shaped_cfg is not None and shaped_cfg.get('enabled', False))
        replay_discount = (
            self._discount - self._discount_alpha - self._discount_beta)
        if self.shaped_reward_enabled:
            if self.cfg.reward_type != 'sparse':
                raise ValueError(
                    'shaped_reward requires reward_type=sparse')
            if (str(shaped_cfg.get('type', 'pbrs')) == 'pbrs'
                    and not math.isclose(
                        float(shaped_cfg.get('pbrs_gamma', replay_discount)),
                        replay_discount,
                        rel_tol=0.0,
                        abs_tol=1.0e-12)):
                raise ValueError(
                    'shaped_reward.pbrs_gamma must equal the replay discount '
                    f'({replay_discount})')
        # create envs
        reward_frame_kwargs = {}
        if shaped_cfg is not None:
            reward_frame_kwargs = {
                'reward_render_size': shaped_cfg.get(
                    'reward_render_size', 224),
                'reward_crop_size': shaped_cfg.get('reward_crop_size', None),
                'reward_crop_offset': shaped_cfg.get(
                    'reward_crop_offset', None),
            }
        self.train_env = mw.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed,
                                  reward_type=self.cfg.reward_type,
                                  **reward_frame_kwargs)
        self.eval_env, self.eval_envs = None, None
        if self.cfg.num_eval_envs > 1:
            self.eval_envs = ParallelMetaWorld(self.cfg.task_name,
                                              self.cfg.frame_stack,
                                              self.cfg.action_repeat,
                                              self.cfg.seed,
                                              self.cfg.num_eval_envs,
                                              reward_type=self.cfg.reward_type)
        else:
            self.eval_env = mw.make(self.cfg.task_name, self.cfg.frame_stack,
                                     self.cfg.action_repeat, self.cfg.seed,
                                     reward_type=self.cfg.reward_type)
        # Start evaluation workers before the shaped-reward model initializes CUDA.
        if self.shaped_reward_enabled:
            self.train_env = MetaWorldShapedRewardWrapper(
                self.train_env, shaped_cfg, str(self.device))
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            math.floor(self._nstep + self._nstep_alpha),
            self._discount - self._discount_alpha - self._discount_beta)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def discount(self):
        return self._discount - self._discount_alpha * math.exp(
            -self.global_step /
            self._discount_alpha_temp) - self._discount_beta * math.exp(
                -self.global_step / self._discount_beta_temp)

    @property
    def nstep(self):
        return math.floor(self._nstep + self._nstep_alpha *
                          math.exp(-self.global_step / self._nstep_alpha_temp))

    def update_buffer(self):
        #self.buffer.update_discount(self.discount)
        self.buffer.update_nstep(self.nstep)
        return

    def eval(self):
        eval_start = time.time()
        if self.eval_envs is not None:
            self._eval_parallel(eval_start)
            return
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            episode_sr = False
            time_step = self.eval_env.reset()
            self.video_recorder.init_obs(time_step.observation,
                                         enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):

                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record_obs(time_step.observation)
                episode_sr = episode_sr or time_step.success
                total_reward += time_step.reward
                step += 1

            total_sr += episode_sr
            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        self._log_eval(step, episode, total_reward, total_sr, eval_start)

    def _eval_parallel(self, eval_start):
        n = self.eval_envs.num_envs
        step, episode, total_reward, total_sr = 0, 0, 0.0, 0.0

        while episode < self.cfg.num_eval_episodes:
            # only the first `keep` envs count when n does not divide the total
            keep = min(n, self.cfg.num_eval_episodes - episode)
            obs, _, _, _ = self.eval_envs.reset()
            self.video_recorder.init_obs(obs[0], enabled=(episode == 0))
            ep_reward = np.zeros(n, dtype=np.float64)
            ep_sr = np.zeros(n, dtype=bool)
            last = np.zeros(n, dtype=bool)
            while not last.any():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act_batch(obs,
                                                  self.global_step,
                                                  eval_mode=True)
                obs, reward, success, last = self.eval_envs.step(action)
                self.video_recorder.record_obs(obs[0])
                ep_sr |= success
                ep_reward += reward
                step += keep

            total_reward += ep_reward[:keep].sum()
            total_sr += ep_sr[:keep].sum()
            episode += keep
            self.video_recorder.save(f'{self.global_frame}.mp4')

        self._log_eval(step, episode, total_reward, total_sr, eval_start)

    def _log_eval(self, step, episode, total_reward, total_sr, eval_start):
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_success_rate', total_sr / episode)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
        print(f'[eval] frame={self.global_frame} '
              f'episodes={episode} took {time.time() - eval_start:.1f}s')

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward, episode_sr = 0, 0, False
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        # self.train_video_recorder.init(time_step.observation)
        metrics = None
        progress_bar = tqdm(total=self.cfg.num_train_frames,
                    initial=self.global_frame,
                    desc='train',
                    unit='frame',
                    dynamic_ncols=True)
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # self.train_video_recorder.save(f'{self.global_frame}.mp4')

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_success_rate', episode_sr)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_sr = False
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step) and self.global_step % self.cfg.update_every_steps == 0:   
                metrics = self.agent.update(
                    self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            if self.shaped_reward_enabled:
                self.logger.log_metrics(
                    self.train_env.last_reward_metrics,
                    self.global_frame,
                    ty='train')
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            #self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            progress_bar.update(self.cfg.action_repeat)

        progress_bar.close()

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfgs):
    from train_mw import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfgs)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    try:
        workspace.train()
    finally:
        if workspace.eval_envs is not None:
            workspace.eval_envs.close()


if __name__ == '__main__':
    main()
