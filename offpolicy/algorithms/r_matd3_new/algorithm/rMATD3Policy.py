from offpolicy.algorithms.r_maddpg.algorithm.rMADDPGPolicy import R_MADDPGPolicy

class R_MATD3Policy(R_MADDPGPolicy):
    def __init__(self, config, policy_config, train=True):
        super(R_MADDPGPolicy, self).__init__(config, policy_config, target_noise=config["target_action_noise_std"], train=train)
