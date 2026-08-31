import metaworld
import random
import os
import sys
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import test_cases_latest_nonoise
from metaworld.data.dataset import *
from datetime import datetime
import gym
import argparse


# Suppress float conversion warnings
gym.logger.set_level(40)


def patch_legacy_time_limit():
    """Keep this old Meta-World code compatible with Gym 0.26's step API."""
    class LegacyTimeLimit(gym.Wrapper):
        def __init__(self, env, max_episode_steps=None):
            super().__init__(env)
            self._max_episode_steps = max_episode_steps
            self._elapsed_steps = 0

        def step(self, action):
            observation, reward, done, info = self.env.step(action)
            self._elapsed_steps += 1
            if self._elapsed_steps >= self._max_episode_steps:
                done = True
            return observation, reward, done, info

        def reset(self, **kwargs):
            self._elapsed_steps = 0
            return self.env.reset(**kwargs)

    gym.wrappers.TimeLimit = LegacyTimeLimit
    import gym.wrappers.time_limit as time_limit
    time_limit.TimeLimit = LegacyTimeLimit


patch_legacy_time_limit()


def crop_box_bounds(image_height, image_width, crop_height, crop_width, crop_offset):
    """Return (top, left, bottom, right) for a centered or translated crop.

    ``crop_offset`` follows the convention used by hdf5_obs_to_mp4.py:
    ``(left, top)`` in pixels from the image borders.
    """
    if crop_offset is None:
        top = (image_height - crop_height) // 2
        left = (image_width - crop_width) // 2
    else:
        left, top = crop_offset
        if left < 0 or top < 0:
            raise ValueError(
                f"Crop offset must be non-negative, got left={left}, top={top}"
            )

    bottom = top + crop_height
    right = left + crop_width
    if bottom > image_height or right > image_width:
        location = "centered" if crop_offset is None else f"left={left}, top={top}"
        raise ValueError(
            f"Crop {crop_height}x{crop_width} at {location} exceeds frame size "
            f"{image_height}x{image_width}"
        )
    return top, left, bottom, right


def configure_image_crop(render_res, crop_size=None, crop_offset=None):
    """Validate a crop and return its bounds plus the saved image resolution."""
    if crop_offset is not None and crop_size is None:
        raise ValueError("--crop-offset requires --center-crop H W")
    if crop_size is None:
        return None, render_res

    crop_height, crop_width = crop_size
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("--center-crop H W must both be positive")
    crop_box = crop_box_bounds(
        render_res[0], render_res[1], crop_height, crop_width, crop_offset
    )
    return crop_box, (crop_height, crop_width)


def crop_frame(frame, crop_box):
    """Crop one RGB or depth frame, copying it before the next env step."""
    if crop_box is None or frame is None:
        return frame
    if frame.ndim < 2:
        raise ValueError(f"Expected a frame with at least 2 dimensions, got {frame.shape}")
    top, left, bottom, right = crop_box
    return frame[top:bottom, left:right].copy()


def gen_data(tasks, num_traj, noise, res, include_depth, camera, data_dir_path,
             write_data=True, write_video=False, video_fps=80,
             max_attempts=None, crop_size=None, crop_offset=None):
    """Generate exactly num_traj successful trajectories per selected task."""
    res = (res, res)
    crop_box, saved_res = configure_image_crop(res, crop_size, crop_offset)
    max_steps_at_goal = 10
    act_tolerance = 1e-5
    lim = 1 - act_tolerance

    print(f'Available tasks: {metaworld.ML1.ENV_NAMES}, in total {len(metaworld.ML1.ENV_NAMES)} tasks.')
    if crop_box is not None:
        location = (
            'centered' if crop_offset is None
            else f'left={crop_offset[0]}, top={crop_offset[1]}'
        )
        print(
            f'Cropping native MuJoCo frames from {res[0]}x{res[1]} to '
            f'{saved_res[0]}x{saved_res[1]} at {location} before writing.'
        )

    for case in test_cases_latest_nonoise:
        if case[0] not in tasks:
            continue

        task_name = case[0]
        policy = case[1]
        print(f'----------Running task {task_name}------------')

        env = metaworld.mw_gym_make(
            task_name,
            goal_cost_reward=False,
            stop_at_goal=True,
            steps_at_goal=max_steps_at_goal,
            cam_height=res[0],
            cam_width=res[1],
            depth=include_depth,
            cam_name=camera,
            train_distrib=True,
        )
        action_space_ptp = env.action_space.high - env.action_space.low

        num_successes = 0
        num_attempts = 0
        task_max_attempts = 10 * num_traj if max_attempts is None else max_attempts
        dt = datetime.now()
        height, width = saved_res
        data_file_name = (
            task_name + '-num-traj_' + str(num_traj) + '-noise_' + str(noise) +
            '-' + dt.strftime("%d-%m-%Y-%H.%M.%S") + '.hdf5'
        )
        video_path_root = 'movies'
        video_dir_path = os.path.join(
            video_path_root,
            task_name + '-noise_' + str(noise) + '-res_' + str(height) +
            '_' + str(width) + '-cam_' + camera + '_' +
            dt.strftime("%d-%m-%Y-%H.%M.%S"),
        )

        data_writer = MWDatasetWriter(
            data_dir_path,
            data_file_name,
            env,
            task_name,
            saved_res,
            camera,
            include_depth,
            act_tolerance,
            max_steps_at_goal,
            write_data=write_data,
        )

        while num_successes < num_traj:
            if task_max_attempts != -1 and num_attempts >= task_max_attempts:
                data_writer.data = data_writer._reset_data()
                data_writer.close()
                raise RuntimeError(
                    f'{task_name}: generated {num_successes}/{num_traj} '
                    f'successful trajectories after {num_attempts} attempts'
                )

            num_attempts += 1
            video_writer = MWVideoWriter(
                video_dir_path,
                task_name + '-' + str(num_attempts),
                video_fps,
                (saved_res[1], saved_res[0]),
                write_video=write_video,
            )

            state = env.reset()
            episode_success = False

            for t in range(env.max_path_length):
                action = policy.get_action(state['full_state'])
                action = np.random.normal(action, noise * action_space_ptp)
                action = np.clip(action, -lim, lim)
                new_state, reward, done, info = env.step(action)
                image = crop_frame(state['image'], crop_box)
                depth = crop_frame(state['depth'], crop_box) if include_depth else state['depth']
                data_writer.append_data(
                    state['full_state'],
                    state['proprio_state'],
                    image,
                    depth,
                    action,
                    reward,
                    done,
                    info,
                )
                video_writer.write(image)
                state = new_state

                if done:
                    episode_success = bool(info.get('task_accomplished', False))
                    if episode_success:
                        print(f'Attempt {num_attempts} succeeded at step {t}')
                    else:
                        print(f'Attempt {num_attempts} failed at time step {t}')
                    break

            if episode_success:
                data_writer.write_trajectory()
                num_successes += 1
            else:
                # Discard the failed episode so it is not written into HDF5.
                data_writer.data = data_writer._reset_data()

        data_writer.close()
        print(f'Generated {num_successes} successful trajectories in {num_attempts} attempts.')
        print(f'Success rate among saved trajectories for {task_name}: 1.0\n')

        if write_data:
            qlearning_dataset(os.path.join(data_dir_path, data_file_name), reward_type='subgoal')


def add_boolean_arg(parser, name, true, false, default):
    assert true.startswith('--') and false.startswith('--')
    assert type(default) is bool
    true_false = parser.add_mutually_exclusive_group()
    true_false.add_argument(true, dest=name, action='store_true')
    true_false.add_argument(false, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tasks", type=str, nargs='+', help="Tasks for which to generate trajectories")
    parser.add_argument("-n", "--num_traj", type=int, help="Number of successful trajectories per task")
    parser.add_argument("-p", "--noise", type=float, default=0, help="Action noise as a fraction of the action space")
    parser.add_argument("-r", "--res", type=int, default=224, help="Native MuJoCo render resolution before optional crop (r x r)")
    parser.add_argument("-c", "--camera", type=str, default='corner2', help="Camera name")
    parser.add_argument("-f", "--video_fps", type=int, default=80, help="Video FPS")
    parser.add_argument("-d", "--data_dir_path", type=str, default='data', help="Directory for demonstration data")
    parser.add_argument(
        "--center-crop",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Crop native frames to HxW before saving; centered unless --crop-offset is set",
    )
    parser.add_argument(
        "--crop-offset",
        type=int,
        nargs=2,
        metavar=("LEFT", "TOP"),
        help="Place the HxW crop with its left/top edges at LEFT/TOP; requires --center-crop",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Maximum attempts per task; default is 10*num_traj, -1 means unlimited",
    )
    add_boolean_arg(parser, 'include_depth', true='--include_depth', false='--noinclude_depth', default=False)
    add_boolean_arg(parser, 'write_data', true='--write_data', false='--nowrite_data', default=True)
    add_boolean_arg(parser, 'write_video', true='--write_video', false='--nowrite_video', default=False)
    args = parser.parse_args()

    if args.center_crop is not None and any(size <= 0 for size in args.center_crop):
        parser.error("--center-crop H W must both be positive")
    if args.crop_offset is not None and args.center_crop is None:
        parser.error("--crop-offset requires --center-crop H W")
    if args.crop_offset is not None and any(offset < 0 for offset in args.crop_offset):
        parser.error("--crop-offset LEFT TOP must both be non-negative")

    print(f'Generating {args.num_traj} successful trajectories with action noise {args.noise} '
          f'for tasks {args.tasks} at {args.res}x{args.res}.')
    gen_data(
        args.tasks,
        args.num_traj,
        args.noise,
        args.res,
        args.include_depth,
        args.camera,
        args.data_dir_path,
        write_data=args.write_data,
        write_video=args.write_video,
        video_fps=args.video_fps,
        max_attempts=args.max_attempts,
        crop_size=tuple(args.center_crop) if args.center_crop is not None else None,
        crop_offset=tuple(args.crop_offset) if args.crop_offset is not None else None,
    )
