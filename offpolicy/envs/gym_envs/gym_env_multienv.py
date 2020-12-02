from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
from envs.MultiAgentEnv import MultiAgentEnv
from flow.utils.registry import make_create_env
from envs.vec_env_wrappers import SubprocVecEnv, DummyVecEnv
import gym

class GymEnvMultiEnv(MultiAgentEnv):
    def __init__(self, env_name):
        self._env = gym.make(env_name).unwrapped

        self._env.reset()
        self.num_agents = 1
        self.agent_ids = [i for i in range(self.num_agents)]

        self.observation_space_dict = {id : self._env.observation_space for id in self.agent_ids}
        self.action_space_dict = {id : self._env.action_space for id in self.agent_ids}
        self.agents = []

    def reset(self):
        return {0: self._env.reset()}

    def step(self, action_dict):
        action = action_dict[0]

        obs, rew, done, info = self._env.step(action)
        obs = {0: obs}
        rew = {0: rew}
        info = {0: info}

        done = {0: done}
        done["env"] = done[0]
        return obs, rew, done, info

    def seed(self, seed):
        self._env.seed(seed)

    def render(self):
        self._env.render()


def make_parallel_env(env_name, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = GymEnvMultiEnv(env_name)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(1)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])