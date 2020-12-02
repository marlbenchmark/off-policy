import numpy as np
import torch
from offpolicy.algorithms.mqmix.algorithm.agent_q_function import AgentQFunction
from torch.distributions import Categorical, OneHotCategorical
from offpolicy.utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose


class M_QMixPolicy:
    def __init__(self, config, policy_config, train=True):
        """
        init relevent args
        """
        self.args = config["args"]
        self.device = config['device']
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete_action = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)
        self.q_network_input_dim = self.obs_dim

        # Local recurrent q network for the agent
        self.q_network = AgentQFunction(
            self.q_network_input_dim, self.act_dim, self.args, self.device)

        if train:
            self.schedule = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish, self.args.epsilon_anneal_time,
                                                  decay="linear")

    def get_q_values(self, observation_batch, action_batch=None):
        """
        Get q values for state action pair batch
        Prev_action_batch: batch_size x action_dim, rows are onehot is onehot row matrix, but action_batch is a nx1 vector (not onehot)
        """
        q_batch = self.q_network(observation_batch)
        if action_batch is not None:
            if type(action_batch) == np.ndarray:
                action_batch = torch.FloatTensor(action_batch)
            if self.multidiscrete:
                all_q_values = []
                for i in range(len(self.act_dim)):
                    curr_q_batch = q_batch[i]
                    curr_action_batch = action_batch[i]
                    curr_q_values = torch.gather(
                        curr_q_batch, 1, curr_action_batch.unsqueeze(dim=-1))
                    all_q_values.append(curr_q_values)
                return torch.cat(all_q_values, dim=-1)
            else:
                q_values = torch.gather(
                    q_batch, 1, action_batch.unsqueeze(dim=-1))
                # q_values is a column vector containing q values for the actions specified by action_batch
                return q_values
        else:
            # if no action specified return all q values
            return q_batch

    def get_actions(self, observation_batch, available_actions=None, t_env=None, explore=False, use_target=None, use_gumbel=None):
        """
        get actions in epsilon-greedy manner, if specified
        """
        batch_size = observation_batch.shape[0]
        q_values = self.get_q_values(observation_batch)
        # mask the available actions by giving -inf q values to unavailable actions
        if available_actions is not None:
            if type(available_actions) == np.ndarray:
                available_actions = torch.FloatTensor(available_actions)
            q_values[available_actions == 0.0] = -1e10
        #greedy_Qs, greedy_actions = list(map(lambda a: a.max(dim=-1), q_values))
        if self.multidiscrete:
            onehot_actions = []
            greedy_Qs = []
            for i in range(len(self.act_dim)):
                greedy_Q, greedy_action = q_values[i].max(dim=-1)
                if explore:
                    eps = self.schedule.eval(t_env)
                    rand_number = np.random.rand(batch_size)
                    # random actions sample uniformly from action space
                    random_action = Categorical(logits=torch.ones(
                        batch_size, self.act_dim[i])).sample()
                    take_random = (rand_number < eps).astype(int)
                    action = (1 - take_random) * greedy_action.numpy() + \
                        take_random * random_action.detach().cpu().numpy()
                    onehot_action = make_onehot(action, self.act_dim[i])
                else:
                    greedy_Q = greedy_Q.unsqueeze(-1)
                    onehot_action = make_onehot(greedy_action, self.act_dim[i])
                onehot_actions.append(onehot_action)
                greedy_Qs.append(greedy_Q)
            onehot_actions = np.concatenate(onehot_actions, axis=-1)
            greedy_Qs = torch.cat(greedy_Qs, dim=-1)
            return onehot_actions, greedy_Qs
        else:
            greedy_Qs, greedy_actions = q_values.max(dim=-1)
            if explore:
                eps = self.schedule.eval(t_env)
                rand_numbers = np.random.rand(batch_size)
                logits = torch.ones(batch_size, self.act_dim)
                random_actions = avail_choose(
                    logits, available_actions).sample()
                take_random = (rand_numbers < eps).astype(int)
                actions = (1 - take_random) * greedy_actions.numpy() + \
                    take_random * random_actions.detach().cpu().numpy()
                return make_onehot(actions, self.act_dim), greedy_Qs
            else:
                greedy_Qs = greedy_Qs.unsqueeze(-1)
                return make_onehot(greedy_actions, self.act_dim), greedy_Qs

    def get_random_actions(self, obs, available_actions=None):
        batch_size = obs.shape[0]
        if available_actions is not None:
            logits = torch.ones(batch_size, self.act_dim)
            random_actions = avail_choose(logits, available_actions)
            random_actions = random_actions.sample()
            random_actions = make_onehot(random_actions, self.act_dim)
        else:
            if self.discrete_action:
                if self.multidiscrete:
                    random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy()
                                      for i in range(len(self.act_dim))]
                    random_actions = np.concatenate(random_actions, axis=-1)
                else:
                    random_actions = OneHotCategorical(logits=torch.ones(
                        batch_size, self.act_dim)).sample().numpy()
            else:
                random_actions = np.random.uniform(self.act_space.low, self.act_space.high,
                                                   size=(batch_size, self.act_dim))
        return random_actions

    def parameters(self):
        return self.q_network.parameters()

    def load_state(self, source_policy):
        self.q_network.load_state_dict(source_policy.q_network.state_dict())
