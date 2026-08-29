"""Parallel MetaWorld envs for evaluation.

Episodes are fixed-length (TimeLimit(250), no early termination), so all envs
reset and finish in lockstep and no autoreset handling is needed.
"""
import atexit
import multiprocessing as mp

import numpy as np


def _worker(conn, task_name, frame_stack, action_repeat, seed, reward_type):
    import metaworld_env as mw

    # MetaWorld's _get_state_rand_vec draws from the global numpy RNG, not the
    # env seed, so workers must be seeded explicitly or they all replay the
    # same task sequence.
    np.random.seed(seed)
    env = mw.make(task_name, frame_stack, action_repeat, seed,
                  reward_type=reward_type)
    conn.send(env.action_spec().shape)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == 'reset':
                ts = env.reset()
                conn.send((ts.observation, ts.reward, ts.success, ts.last()))
            elif cmd == 'step':
                ts = env.step(data)
                conn.send((ts.observation, ts.reward, ts.success, ts.last()))
            elif cmd == 'close':
                return
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        conn.close()


class ParallelMetaWorld:
    """Runs `num_envs` MetaWorld envs in worker processes, one pipe each."""

    def __init__(self, task_name, frame_stack, action_repeat, base_seed,
                 num_envs, reward_type='dense', start_method='forkserver'):
        try:
            ctx = mp.get_context(start_method)
        except ValueError:
            ctx = mp.get_context('spawn')

        self.num_envs = num_envs
        self._conns = []
        self._procs = []
        self._closed = False
        for rank in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker,
                               args=(child, task_name, frame_stack,
                                     action_repeat, base_seed + rank,
                                     reward_type),
                               daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self.action_shape = [c.recv() for c in self._conns][0]
        atexit.register(self.close)

    def _gather(self):
        results = [c.recv() for c in self._conns]
        obs = np.stack([r[0] for r in results])
        reward = np.array([r[1] for r in results], dtype=np.float32)
        success = np.array([bool(r[2]) for r in results])
        last = np.array([bool(r[3]) for r in results])
        return obs, reward, success, last

    def reset(self):
        for c in self._conns:
            c.send(('reset', None))
        return self._gather()

    def step(self, actions):
        # send all before receiving any, otherwise the workers run serially
        for c, a in zip(self._conns, actions):
            c.send(('step', a))
        return self._gather()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for c in self._conns:
            try:
                c.send(('close', None))
                c.close()
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __del__(self):
        self.close()
