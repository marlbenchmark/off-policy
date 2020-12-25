from offpolicy.algorithms.r_maddpg.algorithm.r_actor_critic import R_Actor, R_Critic

class R_MATD3_Actor(R_Actor):
    pass

class R_MATD3_Critic(R_Critic):
    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(R_MATD3_Critic, self).__init__(args, central_obs_dim, central_act_dim, device, num_q_outs=2)
