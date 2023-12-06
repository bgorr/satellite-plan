import keras
from keras import layers
import tensorflow as tf
import math
import random
from math import cos, pi
from collections import namedtuple, deque
import numpy as np





@keras.saving.register_keras_serializable(package="PlanningAndScheduling", name="PlanningCritic")
class PlanningCritic(tf.keras.Model):

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_tracker = tf.keras.metrics.Mean(name="loss")


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Takes a set of N actions from N satellites
        # Each action is an int value [1-5] representing the action taken
        self.num_agents = 3

        self.num_inputs = 0
        for agent in range(self.num_agents):
            self.num_inputs += 2  # agent state inputs
            self.num_inputs += 0  # agent decision value input

        self.input_layer = layers.InputLayer(input_shape=(self.num_inputs,), name="critic_input_layer")

        # Sequence of hidden layers
        self.hidden_layers = keras.Sequential([
            layers.Dense(64, activation="leaky_relu"),
            layers.Dense(64, activation="leaky_relu"),
            layers.Dense(64, activation="leaky_relu"),
        ])

        # Output layer
        # - single value estimating
        self.output_layer = layers.Dense(1, name="mlp_output_layer")
        self.activation = layers.Activation('linear', dtype='float32')

        # Optimizer
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Replay Buffer
        self.buffer_size = 10000
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.norm_values = []
        for i in range(self.num_agents):
            self.norm_values.extend([8640.0, 45.0])  # [8640.0, 45.0, 3.0, 100.0, 100.0]
        self.norm_values = tf.constant(self.norm_values, dtype=tf.float32)


    def implicit_build(self):
        design_input = tf.zeros((1, self.num_inputs))
        self.call(design_input)
        return self


    @tf.function
    def call(self, inputs, training=True, mask=None):
        state = inputs

        # Normalize state
        state = state / self.norm_values

        activations = self.input_layer(state)
        activations = self.hidden_layers(activations)
        activations = self.output_layer(activations)
        prediction = self.activation(activations)

        return prediction


    @tf.function
    def train_step(self, observation_buffer, return_buffer):
        # observation_buffer: (trajectory_len, num_agents * state_dim)
        # return_buffer: (trajectory_len)
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            logits = self.call(observation_buffer)
            loss = self.loss_fn(return_buffer, logits)
        critic_grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return self.loss_tracker.result()



