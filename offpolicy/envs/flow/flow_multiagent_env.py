from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
from envs.MultiAgentEnv import MultiAgentEnv
from flow.utils.registry import make_create_env
from envs.vec_env_wrappers import SubprocVecEnv, DummyVecEnv

class FlowMultiEnv(MultiAgentEnv):
    def __init__(self, flow_params):
        make_env, _ = make_create_env(flow_params)
        self._env = make_env()
        # rand_int = np.random.randint(0, 50)
        # text_file = open("/Users/akashvelu/Documents/names" + str(rand_int) + ".txt", "w")
        # # n = text_file.write(self._env.scenario.name + '\n')
        # text_file.close()

        # self._env.reset()
        self.num_agents = flow_params['env'].additional_params['num_agents'] #TODO: max # agents
        self.agent_ids = [i for i in range(self.num_agents)]

        self.observation_space_dict = {id : self._env.observation_space for id in self.agent_ids}
        self.action_space_dict = {id : self._env.action_space for id in self.agent_ids}
        self.agents = []

    def reset(self):
        # print("OTHER NAME: ", self._env.scenario.name)
        obs = self.convert_dict(self._env.reset())
        return obs

    def step(self, action_dict):
        flow_rl_ids = sorted(list(self._env.get_state().keys()))
        flow_act_dict = {flow_rl_ids[i] : action_dict[i] for i in self.agent_ids}
        obs, rew, done, info = self._env.step(flow_act_dict)
        obs = self.convert_dict(obs)
        rew = self.convert_dict(rew)
        info = self.convert_dict(info)

        env_done = done["__all__"]
        done = self.convert_dict(done)
        done["env"] = env_done
        return obs, rew, done, info

    def seed(self, seed):
        self._env.seed(seed)

    def convert_dict(self, d):
        if '__all__' in d.keys():
            return {i : d['__all__'] for i in self.agent_ids}

        rl_ids = sorted(list(d.keys()))
        if len(rl_ids) != self.num_agents:
            print(rl_ids)
            print(self.num_agents)
        assert len(rl_ids) == self.num_agents, "Number of keys in dict does not match # agents!"
        return {i : d[rl_ids[i]] for i in self.agent_ids}


def make_parallel_env(flow_params, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = FlowMultiEnv(flow_params)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(1)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])