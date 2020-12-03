import numpy as np
import torch
from offpolicy.algorithms.qmix.algorithm.agent_q_function import AgentQFunction
from torch.distributions import Categorical, OneHotCategorical
from offpolicy.utils.util import get_dim_from_space, is_discrete, is_multidiscrete, make_onehot, DecayThenFlatSchedule, avail_choose


class QMixPolicy:
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
        self.hidden_size = self.args.hidden_size
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete_action = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        if self.args.prev_act_inp:
            # this is only local information so the agent can act decentralized
            self.q_network_input_dim = self.obs_dim + self.act_dim
        else:
            self.q_network_input_dim = self.obs_dim

        # Local recurrent q network for the agent
        self.q_network = AgentQFunction(
            self.q_network_input_dim, self.act_dim, self.args, self.device)

        if train:
            self.schedule = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish, self.args.epsilon_anneal_time,
                                                  decay="linear")

    def get_q_values(self, observation_batch, prev_action_batch, hidden_states, action_batch=None):
        """
        Get q values for state action pair batch
        Prev_action_batch: batch_size x action_dim, rows are onehot is onehot row matrix, but action_batch is a nx1 vector (not onehot)
        """
        if len(observation_batch.shape) == 3:
            sequence = True
            batch_size = observation_batch.shape[1]
        else:
            sequence = False
            batch_size = observation_batch.shape[0]

        # combine previous action with observation for input into q, if specified in args
        if self.args.prev_act_inp:
            if type(prev_action_batch) == np.ndarray:
                prev_action_batch = torch.FloatTensor(prev_action_batch)
            input_batch = torch.cat(
                (observation_batch, prev_action_batch), dim=-1)
        else:
            input_batch = observation_batch

        q_batch, new_hidden_batch = self.q_network(input_batch, hidden_states)

        if action_batch is not None:
            if type(action_batch) == np.ndarray:
                action_batch = torch.FloatTensor(action_batch)
            if self.multidiscrete:
                ind = 0
                all_q_values = []
                for i in range(len(self.act_dim)):
                    curr_q_batch = q_batch[i]
                    curr_action_portion = action_batch[:,:, ind: ind + self.act_dim[i]]
                    curr_action_inds = curr_action_portion.max(dim=-1)[1]
                    curr_q_values = torch.gather(
                        curr_q_batch, 2, curr_action_inds.unsqueeze(dim=-1))
                    all_q_values.append(curr_q_values)
                    ind += self.act_dim[i]
                return torch.cat(all_q_values, dim=-1), new_hidden_batch
            else:
                # convert one-hot action batch to index tensors to gather the q values corresponding to the actions taken
                action_batch = action_batch.max(dim=-1)[1]
                # import pdb; pdb.set_trace()
                q_values = torch.gather(
                    q_batch, 2, action_batch.unsqueeze(dim=-1))
                # q_values is a column vector containing q values for the actions specified by action_batch
                return q_values, new_hidden_batch
        else:
            # if no action specified return all q values
            return q_batch, new_hidden_batch

    def get_actions(self, observation_batch, prev_action_batch, hidden_states, available_actions=None, t_env=None, explore=False, use_target=None, use_gumbel=None):
        """
        get actions in epsilon-greedy manner, if specified
        """
        if len(observation_batch.shape) == 2:
            batch_size = observation_batch.shape[0]
            no_sequence = True
        else:
            batch_size = observation_batch.shape[1]
            seq_len = observation_batch.shape[0]
            no_sequence = False

        q_values_out, new_hidden_states = self.get_q_values(
            observation_batch, prev_action_batch, hidden_states)

        # mask the available actions by giving -inf q values to unavailable actions
        if available_actions is not None:
            if type(available_actions) == np.ndarray:
                available_actions = torch.FloatTensor(available_actions)
            q_values = q_values_out.clone()
            q_values[available_actions == 0.0] = -1e10
        else:
            q_values = q_values_out
        #greedy_Qs, greedy_actions = list(map(lambda a: a.max(dim=-1), q_values))
        if self.multidiscrete:
            onehot_actions = []
            greedy_Qs = []
            for i in range(len(self.act_dim)):
                greedy_Q, greedy_action = q_values[i].max(dim=-1)

                if explore:
                    assert no_sequence, "Can only explore on non-sequences"
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
                    if no_sequence:
                        onehot_action = make_onehot(
                            greedy_action, self.act_dim[i])
                    else:
                        onehot_action = make_onehot(
                            greedy_action, self.act_dim[i], seq_len=seq_len)

                onehot_actions.append(onehot_action)
                greedy_Qs.append(greedy_Q)

            onehot_actions = np.concatenate(onehot_actions, axis=-1)
            greedy_Qs = torch.cat(greedy_Qs, dim=-1)

            if explore:
                return onehot_actions, new_hidden_states.detach(), greedy_Qs
            else:
                return onehot_actions, new_hidden_states, greedy_Qs
        else:
            greedy_Qs, greedy_actions = q_values.max(dim=-1)
            if explore:
                assert no_sequence, "Can only explore on non-sequences"
                eps = self.schedule.eval(t_env)
                rand_numbers = np.random.rand(batch_size)
                logits = torch.ones(batch_size, self.act_dim)
                random_actions = avail_choose(
                    logits, available_actions).sample()
                take_random = (rand_numbers < eps).astype(int)
                actions = (1 - take_random) * greedy_actions.numpy() + \
                    take_random * random_actions.detach().cpu().numpy()
                return make_onehot(actions, self.act_dim), new_hidden_states.detach(), greedy_Qs

            else:
                greedy_Qs = greedy_Qs.unsqueeze(-1)
                if no_sequence:
                    return make_onehot(greedy_actions, self.act_dim), new_hidden_states, greedy_Qs
                else:
                    return make_onehot(greedy_actions, self.act_dim, seq_len=seq_len), new_hidden_states, greedy_Qs

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

    def init_hidden(self, num_agents, batch_size, use_numpy=False):
        if use_numpy:
            if num_agents == -1:
                return np.zeros((batch_size, self.hidden_size))
            else:
                return np.zeros((num_agents, batch_size, self.hidden_size))
        else:
            if num_agents == -1:
                return torch.zeros(batch_size, self.hidden_size)
            else:
                return torch.zeros(num_agents, batch_size, self.hidden_size)

    def parameters(self):
        return self.q_network.parameters()

    def load_state(self, source_policy):
        self.q_network.load_state_dict(source_policy.q_network.state_dict())
