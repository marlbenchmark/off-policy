"""Example of using a custom RNN keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class FeedForward(TFModelV2):
    """Simple custom gated recurrent unit."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,):
        super(FeedForward, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)

        # Define input layers
        if 'original_space' in dir(obs_space):
            curr_obs_space = obs_space.original_space.spaces["obs"]
        else:
            curr_obs_space = obs_space
        self.use_prev_action = model_config["custom_options"].get("use_prev_action")
        if self.use_prev_action:
            obs_shape = curr_obs_space.shape[0]
            action_shape = action_space.shape[0]
            input_layer = tf.keras.layers.Input(
                shape=(obs_shape + action_shape), name="inputs")
        else:
            input_layer = tf.keras.layers.Input(
                shape=(curr_obs_space.shape[0]), name="inputs")
        # Preprocess observations with the appropriate number of hidden layers
        last_layer = input_layer
        i = 1
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1

        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(last_layer)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(last_layer)

        inputs = [input_layer]

        # Create the RNN model
        self.model = tf.keras.Model(
            inputs=inputs,
            outputs=[logits, values])
        self.register_variables(self.model.variables)
        self.model.summary()

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn()"""
        # first we add the time dimension for each object
        if isinstance(input_dict["obs"], dict):
            obs = input_dict["obs"]["obs"]
        else:
            obs = input_dict["obs"]

        if self.use_prev_action:
            action = input_dict["prev_actions"]
            obs = tf.concat([obs, action], axis=-1)

        model_out, self._value_out = self.model([obs])
        return model_out, state
