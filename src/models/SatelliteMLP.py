import keras
from keras import layers
import tensorflow as tf
import math
import random
from math import cos, pi
from collections import namedtuple, deque


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@keras.saving.register_keras_serializable(package="SatelliteMLP", name="SatelliteMLP")
class SatelliteMLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----- STATE SPACE -----
        # 1. Satellite mission_time
        # 2. Satellite slewing angle
        self.state_vars = 2
        self.input_layer = layers.InputLayer(input_shape=(self.state_vars,), name="mlp_input_layer")

        # Hidden Layers
        self.hidden_0 = layers.Dense(16, activation="leaky_relu", name="mlp_hidden_0")
        self.hidden_1 = layers.Dense(32, activation="leaky_relu", name="mlp_hidden_1")
        self.hidden_2 = layers.Dense(16, activation="leaky_relu", name="mlp_hidden_2")

        # Output Layers
        # - probability distribution over possible actions
        self.total_actions = 10
        self.output_layer = layers.Dense(self.total_actions, name="mlp_output_layer")
        self.activation = layers.Activation('linear', dtype='float32')

        # Optimizer
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Hyperparameters
        self.epsilon = 0.99
        self.step = 1

        # Replay Buffer
        self.buffer_size = 1000
        self.replay_buffer = deque(maxlen=self.buffer_size)


    def call(self, inputs, training=True, mask=None):
        state = inputs

        activations = self.input_layer(state)
        activations = self.hidden_0(activations)
        activations = self.hidden_1(activations)
        activations = self.hidden_2(activations)
        activations = self.output_layer(activations)
        probabilities = self.activation(activations)

        return probabilities

    def linear_decay(self):
        epsilon_end = 0.15
        total_steps = 1000
        if self.step > total_steps:
            return epsilon_end
        return self.epsilon - self.step * (self.epsilon - epsilon_end) / total_steps

    def cosine_decay(self):
        decay_steps = 800
        decay_min = 0.15
        step = min(self.step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - decay_min) * cosine_decay + decay_min
        return self.epsilon * decayed

    def implicit_build(self):
        design_input = tf.zeros((1, self.state_vars))
        self.call(design_input)
        return self

    def get_state_tensor(self, state):
        state_tensor = tf.convert_to_tensor(state)           # State is list of state variables
        state_tensor = tf.expand_dims(state_tensor, axis=0)  # Add batch dimension
        return state_tensor

    def get_aciton(self, state, training=True, num_actions=None, debug=False):
        if num_actions is None:
            num_actions = self.total_actions

        # Get state tensor
        state_tensor = self.get_state_tensor(state)

        # Forward pass
        q_values = self.call(state_tensor, training=training)
        q_values = q_values[0, :num_actions]

        # Either random action or greedy action (epsilon greedy)
        # epsilon = self.cosine_decay()
        epsilon = self.linear_decay()
        if tf.random.uniform(shape=()) < epsilon:
            action_idx = tf.random.uniform(shape=(), minval=0, maxval=num_actions, dtype=tf.int64)
        else:
            action_idx = tf.argmax(q_values)
        self.step += 1

        # Debug
        if debug is True:
            q_list = q_values.numpy().tolist()
            q_list = [round(q, 2) for q in q_list]
            print('STEP, EPSILON, Q-VALUES:', self.step, round(epsilon, 2), '|', q_list)

        return action_idx.numpy()

    def get_q_value(self, state, action_idx=None, training=True):

        # Get state tensor
        state_tensor = self.get_state_tensor(state)

        # Forward pass
        q_values = self.call(state_tensor, training=training)
        q_values = q_values[0, :]
        if action_idx:
            q_value = tf.gather(q_values, action_idx)
        else:
            max_idx = tf.argmax(q_values)
            q_value = tf.gather(q_values, max_idx)
        return q_value

    def get_q_value_batch(self, states, action_idxs=None, training=True):

        # Get state tensor
        state_tensor = tf.convert_to_tensor(states)

        # Forward pass
        q_values = self.call(state_tensor, training=training)

        if action_idxs is not None:
            # If action indices are provided, gather Q values based on these indices
            q_values = tf.gather(q_values, action_idxs, batch_dims=1)
        else:
            # If not, take max Q value for each state in the batch
            q_values = tf.reduce_max(q_values, axis=1)

        return q_values

    # -------------------------------------------
    # Set weights
    # -------------------------------------------

    def load_target_weights(self, model_instance, trainable=False):
        """
        Updates the weights of the current instance with the weights of the provided model instance.
        Args:
            model_instance (SatelliteMLP): Instance of SatelliteMLP whose weights will be used.
        """
        for target_layer, source_layer in zip(self.layers, model_instance.layers):
            target_layer.set_weights(source_layer.get_weights())
        self.trainable = trainable

    # -------------------------------------------
    # Experience Replay
    # -------------------------------------------

    def record_experience(self, state, action, reward, next_state, done=False):
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

    def sample_buffer(self, batch_size, indices=None):
        if indices is None:
            indices = random.sample(range(len(self.replay_buffer)), batch_size)
        return [self.replay_buffer[i] for i in indices]

    # -------------------------------------------
    # Not used
    # -------------------------------------------

    # def update_model(self, state, action_idx, reward, training=True):
    #
    #     # Expand to give batch dimension
    #     state_tensor = self.get_state_tensor(state)
    #
    #     with tf.GradientTape() as tape:
    #
    #         probs = self.call(state_tensor, training=training)
    #         probs = probs[0, :]
    #         action_prob = tf.gather(probs, action_idx)
    #
    #         loss = -tf.math.log(action_prob) * reward
    #
    #     # Calculate gradients
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #
    #     # Apply gradients
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #
    #     return loss




















    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

























