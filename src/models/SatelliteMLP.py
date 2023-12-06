import keras
from keras import layers
import tensorflow as tf
import math
import random
from math import cos, pi
from collections import namedtuple, deque
import numpy as np


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@keras.saving.register_keras_serializable(package="SatelliteMLP", name="SatelliteMLP")
class SatelliteMLP(tf.keras.Model):

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_tracker = tf.keras.metrics.Mean(name="loss")
    kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----- STATE SPACE -----
        # 1. Satellite mission_time: 0-8640
        # 2. Satellite slewing angle:
        # 3. Satellite latitude
        # 4. Satellite longitude
        self.state_vars = 2
        self.input_layer = layers.InputLayer(input_shape=(self.state_vars,), name="mlp_input_layer")

        # Epsilon Greedy
        self.epsilon_start = 0.99
        self.epsilon_end = 0.005
        self.decay_steps = 200000

        # sequence of hidden layers
        self.hidden_layers = keras.Sequential([
            layers.Dense(64, activation="leaky_relu"),
            layers.Dense(64, activation="leaky_relu"),
            layers.Dense(64, activation="leaky_relu"),
        ])

        # Output Layers
        # - probability distribution over possible actions
        self.total_actions = 5  # + 1  # 5 obs actions + 1 dl action
        self.dl_action_idx = 5
        self.output_layer = layers.Dense(self.total_actions, name="mlp_output_layer")
        self.activation = layers.Activation('linear', dtype='float32')

        # Optimizer
        self.learning_rate = 0.0005  # PPO 0.00005 | VDN ??? ||| P1 formulation ~
        self.gradient_clip = 0.1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipvalue=self.gradient_clip)

        # Hyperparameters
        self.epsilon = 0.9
        self.step = 1

        # Replay Buffer
        self.buffer_size = 50000
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.memory = []

        # Override Methods
        self.fifo = False

    def implicit_build(self):
        design_input = tf.zeros((1, self.state_vars))
        self.call(design_input)
        return self

    def call(self, inputs, training=True, mask=None):
        state = inputs  # shape = (batch_size, state_vars) --> 2 state vars, time and angle
        state = tf.cast(state, dtype=tf.float32)

        norm_values = tf.constant([8640.0, 45.0], dtype=tf.float32)
        # norm_values = tf.constant([8640.0, 45.0, 3.0, 100.0, 100.0], dtype=tf.float32)
        # norm_values = tf.constant([8649.0, 10000.0, 45.0, 1.0], dtype=tf.float32)
        state = state / norm_values

        activations = self.input_layer(state)
        activations = self.hidden_layers(activations)
        activations = self.output_layer(activations)
        # probabilities = self.activation(activations)

        return activations

    def get_state_tensor(self, state):
        state_tensor = tf.convert_to_tensor(state)           # State is list of state variables
        state_tensor = tf.expand_dims(state_tensor, axis=0)  # Add batch dimension
        return state_tensor

    # -------------------------------------------
    # Epsilon Annealing
    # -------------------------------------------

    def linear_decay(self):
        if self.step > self.decay_steps:
            return self.epsilon_end
        return self.epsilon - self.step * (self.epsilon - self.epsilon_end) / self.decay_steps

    def cosine_decay(self):
        step = min(self.step, self.decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.epsilon_end) * cosine_decay + self.epsilon_end
        return self.epsilon * decayed

    # -------------------------------------------
    # Training / Inference
    # -------------------------------------------

    def get_ppo_action(self, state, num_actions=None):
        if num_actions is None:
            num_actions = self.total_actions

        # Get state tensor
        state_tensor = self.get_state_tensor(state)

        # Forward pass
        logits = self.call(state_tensor)
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)  # Store these for later use
        # print('PPO ACTION:', q_values)
        probs = probs[0, :num_actions]

        top_n_values, top_n_indices = tf.nn.top_k(probs, k=num_actions)
        top_n_probs = top_n_values / tf.reduce_sum(top_n_values)  # renormalize
        action_idx = np.random.choice(top_n_indices.numpy(), p=top_n_probs.numpy())

        self.step += 1
        if self.fifo is True:
            return 0
        else:
            log_prob = log_probs[0, action_idx].numpy()
            return [action_idx, log_prob]

    def get_action(self, state, training=True, num_actions=None, debug=False, rand_action=False, init_action=False):
        if num_actions is None:
            num_actions = self.total_actions

        # Get state tensor
        state_tensor = self.get_state_tensor(state)

        # Forward pass
        q_values = self.call(state_tensor, training=training)
        q_values = q_values[0, :num_actions]

        # Either random action or greedy action (epsilon greedy)
        epsilon = self.linear_decay()
        # epsilon = self.cosine_decay()
        if rand_action is True:
            epsilon = 1
        if tf.random.uniform(shape=()) < epsilon:
            action_idx = tf.random.uniform(shape=(), minval=0, maxval=num_actions, dtype=tf.int64)
        else:
            action_idx = tf.argmax(q_values)


        # Debug
        if debug is True:
            q_list = q_values.numpy().tolist()
            q_list = [round(q, 2) for q in q_list]
            s_list = [round(s, 2) for s in state]
            print('STEP', self.step, 'ACTION', action_idx, 'EPSILON', round(epsilon, 2), 'Q-VALUES', q_list, 'STATE', s_list, 'TIME', state[0])

        self.step += 1
        if self.fifo is True:
            return 0
        else:
            return [action_idx, q_values[action_idx].numpy()]

    @tf.function
    def get_q_value_batch(self, states, action_idxs=None, training=True):
        print('--> TRACING get_q_value_batch_max')

        # Get state tensor: (batch_size, state_vars)
        state_tensor = tf.convert_to_tensor(states)

        # Forward pass: (batch_size, num_actions)
        q_values = self.call(state_tensor, training=training)

        if action_idxs is not None:
            # If action indices are provided, gather Q values based on these indices
            q_values = tf.gather(q_values, action_idxs, batch_dims=1)
        else:
            # If not, take max Q value for each state in the batch
            q_values = tf.reduce_max(q_values, axis=1)

        return q_values

    @tf.function
    def get_q_value_batch_max(self, states):
        print('--> TRACING get_q_value_batch_max')
        state_tensor = tf.convert_to_tensor(states)
        q_values = self.call(state_tensor)      # shape = (batch_size, num_actions)
        return tf.reduce_max(q_values, axis=1)  # shape = (batch_size,)

    @tf.function
    def get_q_idx_batch_max(self, states):
        print('--> TRACING get_q_idx_batch_max')
        state_tensor = tf.convert_to_tensor(states)
        q_values = self.call(state_tensor)  # shape = (batch_size, num_actions)
        return tf.argmax(q_values, axis=1)  # shape = (batch_size,)

    @tf.function
    def get_q_value_batch_idx(self, states, action_idxs):
        print('--> TRACING get_q_value_batch_idx')
        state_tensor = tf.convert_to_tensor(states)
        q_values = self.call(state_tensor)
        return tf.gather(q_values, action_idxs, batch_dims=1)

    @tf.function
    def train_step(self, batch_states, batch_actions, target_q_value, current_q_values):
        print('--> TRACING TRAIN')
        # current_q_values: (batch_size,)
        # target_q_value: (batch_size,)

        with tf.GradientTape() as tape:
            q_values = self.call(batch_states, training=True)           # shape = (batch_size, num_actions)
            q_values = tf.gather(q_values, batch_actions, batch_dims=1) # shape = (batch_size,)

            # Cumulative q values
            q_sum = q_values + current_q_values

            # Loss
            # loss = tf.keras.losses.Huber()(target_q_value, q_sum)
            loss = self.loss_fn(target_q_value, q_sum)

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @tf.function
    def train_step_ppo(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        # print('observation_buffer', observation_buffer.shape)
        # print('action_buffer', action_buffer.shape)
        # print('logprobability_buffer', logprobability_buffer.shape)
        # print('advantage_buffer', advantage_buffer.shape)

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            logits = self.call(observation_buffer)
            probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)  # (batch_size, actions)
            log_probs_act = tf.reduce_sum(
                tf.one_hot(action_buffer, self.total_actions) * log_probs, axis=1
            )

            ratio = tf.exp(log_probs_act - logprobability_buffer)
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + 0.2) * advantage_buffer,
                (1 - 0.2) * advantage_buffer,
            )

            # Calculate entropy
            entropy = -tf.reduce_sum(probs * log_probs, axis=1)
            entropy = tf.reduce_mean(entropy)



            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(policy_grads, self.trainable_variables))

        logits = self.call(observation_buffer)
        log_probs = tf.nn.log_softmax(logits)  # (batch_size, actions)
        log_probs_act = tf.reduce_sum(
            tf.one_hot(action_buffer, self.total_actions) * log_probs, axis=1
        )
        kl = tf.reduce_mean(
            logprobability_buffer - log_probs_act
        )
        kl = tf.reduce_sum(kl)

        # Update metrics
        self.loss_tracker.update_state(policy_loss)
        self.kl_loss_tracker.update_state(kl)

        return self.kl_loss_tracker.result(), self.loss_tracker.result(), entropy



    # -------------------------------------------
    # Load weights
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
        self.memory.append(state)
        self.replay_buffer.append(experience)

    def sample_buffer(self, batch_size, indices=None):
        if indices is None:
            indices = random.sample(range(len(self.replay_buffer)), batch_size)
        return [self.replay_buffer[i] for i in indices]

    def get_memory_time_values(self):
        return [mem[0] for mem in self.memory]





    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

























