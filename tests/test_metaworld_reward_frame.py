import numpy as np
from types import SimpleNamespace

import pytest

import metaworld_env as mw


class _FakeSim:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def render(self, width, height, *, mode, camera_name):
        self.calls.append((width, height, mode, camera_name))
        return self.frame


def test_reward_frame_is_rgb_side_channel():
    env = mw.make(
        "assembly",
        frame_stack=3,
        action_repeat=2,
        seed=7,
        reward_type="sparse",
    )
    time_step = env.reset()
    policy_observation = time_step.observation.copy()

    reward_frame = env.get_reward_frame("corner2")

    assert reward_frame.shape == (224, 224, 3)
    assert reward_frame.dtype == np.uint8
    assert reward_frame.flags.c_contiguous
    np.testing.assert_array_equal(time_step.observation, policy_observation)
    assert time_step.observation.shape == (9, 84, 84)

    next_step = env.step(np.zeros(env.action_spec().shape, dtype=np.float32))
    next_reward_frame = env.get_reward_frame("corner2")
    assert next_step.observation.shape == (9, 84, 84)
    assert next_reward_frame.shape == (224, 224, 3)


def test_reward_frame_crop_uses_training_data_offset_convention():
    full_frame = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    sim = _FakeSim(full_frame)
    env = mw.MetaWorld.__new__(mw.MetaWorld)
    env._env = SimpleNamespace(sim=sim)
    env._camera = "corner2"
    env._reward_render_size, env._reward_crop_box = (
        mw.configure_reward_frame_crop(256, (224, 224), (16, 8)))

    reward_frame = env.get_reward_frame()

    assert sim.calls == [(256, 256, "offscreen", "corner2")]
    assert reward_frame.shape == (224, 224, 3)
    assert reward_frame.dtype == np.uint8
    assert reward_frame.flags.c_contiguous
    np.testing.assert_array_equal(reward_frame, full_frame[8:232, 16:240])


@pytest.mark.parametrize(
    "crop_size, crop_offset, match",
    [
        (None, (0, 0), "requires reward_crop_size"),
        ((224, 224), (-1, 0), "non-negative"),
        ((257, 224), None, "exceeds"),
    ],
)
def test_reward_frame_crop_settings_are_validated(crop_size, crop_offset, match):
    with pytest.raises(ValueError, match=match):
        mw.configure_reward_frame_crop(256, crop_size, crop_offset)
