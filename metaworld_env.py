import os
import operator
import gym
import numpy as np
from dm_env import StepType, specs
import dm_env
import numpy as np
from gym import spaces
from typing import Any, NamedTuple
from collections import deque


def _integer_pair(value, name):
    try:
        first, second = value
        return operator.index(first), operator.index(second)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a two-element integer sequence, got {value!r}"
        ) from error


def configure_reward_frame_crop(render_size, crop_size=None, crop_offset=None):
    """Validate reward-frame crop settings and return ``(top, left, bottom, right)``.

    ``crop_size`` is ``(height, width)`` and ``crop_offset`` is ``(left, top)``,
    matching ``training_data_gen.py``.
    """
    try:
        render_size = operator.index(render_size)
    except TypeError as error:
        raise ValueError(
            f"reward_render_size must be an integer, got {render_size!r}"
        ) from error
    if render_size <= 0:
        raise ValueError("reward_render_size must be positive")
    if crop_offset is not None and crop_size is None:
        raise ValueError("reward_crop_offset requires reward_crop_size")
    if crop_size is None:
        return render_size, None

    crop_height, crop_width = _integer_pair(crop_size, "reward_crop_size")
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("reward_crop_size values must both be positive")
    if crop_offset is None:
        top = (render_size - crop_height) // 2
        left = (render_size - crop_width) // 2
    else:
        left, top = _integer_pair(crop_offset, "reward_crop_offset")
        if left < 0 or top < 0:
            raise ValueError(
                f"reward_crop_offset values must both be non-negative, got "
                f"{(left, top)}")
    bottom = top + crop_height
    right = left + crop_width
    if top < 0 or left < 0 or bottom > render_size or right > render_size:
        location = "centered" if crop_offset is None else (
            f"left={left}, top={top}")
        raise ValueError(
            f"Reward crop {crop_height}x{crop_width} at {location} exceeds "
            f"the {render_size}x{render_size} rendered frame")
    return render_size, (top, left, bottom, right)


class MetaWorld:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        reward_render_size=224,
        reward_crop_size=None,
        reward_crop_offset=None,
    ):
        import metaworld
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )

        os.environ["MUJOCO_GL"] = "egl"

        task = f"{name}-v2-goal-observable"
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_cls(seed=seed)
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat

        self._camera = camera
        self._reward_render_size, self._reward_crop_box = (
            configure_reward_frame_crop(
                reward_render_size, reward_crop_size, reward_crop_offset))

    @property
    def obs_space(self):
        spaces = {
            "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def get_reward_frame(self, camera_name=None):
        camera_name = camera_name or self._camera
        frame = self._env.sim.render(
            self._reward_render_size,
            self._reward_render_size,
            mode="offscreen",
            camera_name=camera_name,
        )
        expected_shape = (
            self._reward_render_size, self._reward_render_size, 3)
        if frame.shape != expected_shape:
            raise ValueError(
                f"Expected a {expected_shape} RGB reward frame, got {frame.shape}")
        if self._reward_crop_box is not None:
            top, left, bottom, right = self._reward_crop_box
            frame = frame[top:bottom, left:right]
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action["action"])
            success += float(info["success"])
            reward += rew or 0.0
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": success,
        }
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        state = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "image": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ),
            "state": state,
            "success": False,
        }
        return obs


class OuterTransition(NamedTuple):
    state: Any
    action: Any
    next_state: Any
    dense_reward: Any
    success: Any


def custom_outer_reward(transition):
    raise NotImplementedError(
        'Implement custom_outer_reward() before using reward_type=custom')


class OuterRewardWrapper:
    _VALID_REWARD_TYPES = ('dense', 'sparse', 'custom')

    def __init__(self, env, reward_type='dense'):
        if reward_type not in self._VALID_REWARD_TYPES:
            raise ValueError(
                f'Unknown reward_type {reward_type!r}; expected one of '
                f'{self._VALID_REWARD_TYPES}')
        self._env = env
        self._reward_type = reward_type
        self._state = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def reset(self):
        obs = self._env.reset()
        self._state = np.array(obs['state'], copy=True)
        return obs

    def step(self, action):
        if self._state is None:
            raise RuntimeError('Must reset environment before stepping.')

        obs = self._env.step(action)
        transition = OuterTransition(
            state=self._state,
            action=np.array(action['action'], copy=True),
            next_state=np.array(obs['state'], copy=True),
            dense_reward=obs['reward'],
            success=bool(obs['success']))

        if self._reward_type == 'sparse':
            obs['reward'] = float(transition.success)
        elif self._reward_type == 'custom':
            obs['reward'] = float(custom_outer_reward(transition))

        self._state = transition.next_state
        return obs


class NormalizeAction:
    def __init__(self, env, key="action"):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})

class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    success: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
class metaworld_wrapper():
    def __init__(self, env, nstack=3):
        self._env = env
        self.nstack = 3
        wos = env.obs_space['image']  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)

        self.observation_space = spaces.Box(low=np.transpose(low, (2, 0, 1)), high=np.transpose(high, (2, 0, 1)), dtype=np.uint8)


    def observation_spec(self):
        return specs.BoundedArray(self.observation_space.shape,
                                  np.uint8,
                                  0,
                                  255,
                                  name='observation')

    def action_spec(self):
        return specs.BoundedArray(self._env.act_space['action'].shape,
                                  np.float32,
                                  self._env.act_space['action'].low,
                                  self._env.act_space['action'].high,
                                  'action')

    def get_reward_frame(self, camera_name=None):
        return self._env.get_reward_frame(camera_name)

    def reset(self):
        time_step = self._env.reset()
        obs = time_step['image']
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                                 step_type=StepType.FIRST,
                                 action=np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype),
                                 reward=0.0,
                                 discount=1.0,
                                success = time_step['success'])
    def step(self, action):
        action = {'action':action}
        time_step = self._env.step(action)
        obs = time_step['image']
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1) #
        self.stackedobs[..., -obs.shape[-1]:] = obs

        if time_step['is_first']:
            step_type = StepType.FIRST
        elif time_step['is_last']:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return ExtendedTimeStep(observation=np.transpose(self.stackedobs, (2, 0, 1)),
                                 step_type=step_type,
                                 action=action['action'],
                                 reward=time_step['reward'],
                                 discount=1.0,
                                success = time_step['success'])

def make(name, frame_stack, action_repeat, seed, reward_type='dense',
         reward_render_size=224, reward_crop_size=None,
         reward_crop_offset=None):
    env = MetaWorld(name, seed, action_repeat, (84, 84), 'corner2',
                    reward_render_size=reward_render_size,
                    reward_crop_size=reward_crop_size,
                    reward_crop_offset=reward_crop_offset)
    env = NormalizeAction(env)
    env = OuterRewardWrapper(env, reward_type)
    env = TimeLimit(env, 250)
    env = metaworld_wrapper(env, frame_stack)

    return env
