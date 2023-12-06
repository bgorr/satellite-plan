import keras
from keras import layers
import tensorflow as tf
import math
import random
from math import cos, pi
from collections import namedtuple, deque
import numpy as np





@keras.saving.register_keras_serializable(package="PlanningAndScheduling", name="MixingNetwork")
class MixingNetwork(tf.keras.Model):

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_tracker = tf.keras.metrics.Mean(name="loss")


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Takes a set of N q-values from N satellite agents
        self.num_agents = 36
        self.num_inputs = 0
        for agent in range(self.num_agents):
            self.num_inputs += 1  # agent q-value
        self.input_layer = layers.InputLayer(input_shape=(self.num_agents,), name="mix_input_layer")

        # Sequence of hidden layers
        self.hidden_layers = keras.Sequential([
            layers.Dense(128, activation="leaky_relu"),
            layers.Dense(128, activation="leaky_relu"),
            layers.Dense(128, activation="leaky_relu"),
        ])

        # Output layer
        # - combined q-value
        self.output_layer = layers.Dense(1, name="mix_output_layer")
        self.activation = layers.Activation('linear', dtype='float32')

        # Optimizer
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    def implicit_build(self):
        design_input = tf.zeros((1, self.num_inputs))
        self.call(design_input)
        return self

    def call(self, inputs, training=True, mask=None):
        state = inputs
        activations = self.input_layer(state)
        activations = self.hidden_layers(activations)
        activations = self.output_layer(activations)
        prediction = self.activation(activations)
        return prediction


    @tf.function
    def train_step(self, inputs):
        print('--> TRACING MIXING NETWORK TRAIN')


