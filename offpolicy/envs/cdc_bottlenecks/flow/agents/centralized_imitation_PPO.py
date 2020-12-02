from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""An example of customizing PPO to leverage a centralized critic with an imitation loss"""

import argparse
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_policy import LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing
import tensorflow as tf

from flow.agents.custom_ppo import AttributeMixin, CustomPPOTrainer
from flow.agents.ImitationPPO import update_kl, ImitationLearningRateSchedule, imitation_loss, loss_stats
from flow.agents.centralized_PPO import CentralizedValueMixin, \
    centralized_critic_postprocessing, loss_with_central_critic

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=100000)


def new_ppo_surrogate_loss(policy, model, dist_class, train_batch):
    policy.imitation_loss = imitation_loss(policy, model, dist_class, train_batch)
    loss = loss_with_central_critic(policy, model, dist_class, train_batch)
    return policy.policy_weight * loss + policy.imitation_weight * policy.imitation_loss


def setup_mixins(policy, obs_space, action_space, config):
    AttributeMixin.__init__(policy, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ImitationLearningRateSchedule.__init__(policy, config["model"]["custom_options"]["num_imitation_iters"],
                                           config["model"]["custom_options"]["imitation_weight"], config)
    # hack: put in a noop VF so some of the inherited PPO code runs
    policy.value_function = tf.zeros(
        tf.shape(policy.get_placeholder(SampleBatch.CUR_OBS))[0])


def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.global_norm(grads),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.central_value_function),
    }


ImitationCentralizedPolicy = PPOTFPolicy.with_updates(
    name="ImitationCentralizedPolicy",
    before_loss_init=setup_mixins,
    postprocess_fn=centralized_critic_postprocessing,
    stats_fn=loss_stats,
    grad_stats_fn=grad_stats,
    loss_fn=new_ppo_surrogate_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin, ImitationLearningRateSchedule
    ])

ImitationCentralizedTrainer = CustomPPOTrainer.with_updates(name="ImitationCentralizedPPOTrainer",
                                                      default_policy=ImitationCentralizedPolicy,
                                                      after_optimizer_step=update_kl)