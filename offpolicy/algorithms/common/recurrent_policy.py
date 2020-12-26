from abc import ABC, abstractmethod


class RecurrentPolicy(ABC):
    @abstractmethod
    def get_actions(self, obs, prev_actions, rnn_states, available_actions, t_env, explore):
        raise NotImplementedError

    @abstractmethod
    def get_random_actions(self, obs, available_actions):
        raise NotImplementedError

    @abstractmethod
    def init_hidden(self, num_agents, batch_size, use_numpy):
        raise NotImplementedError