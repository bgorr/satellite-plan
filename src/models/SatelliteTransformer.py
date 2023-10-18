import keras
from keras import layers
import tensorflow as tf
import math
import random
from math import cos, pi
from collections import namedtuple, deque
import numpy as np
from copy import deepcopy
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding

import config

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@keras.saving.register_keras_serializable(package="SatTransformer", name="SatTransformer")
class SatTransformer(tf.keras.Model):

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_tracker = tf.keras.metrics.Mean(name="loss")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----- STATE SPACE -----
        # 1. Satellite mission_time
        # 2. Satellite slewing angle
        # 3. Satellite latitude
        # 4. Satellite longitude
        self.state_vars = 4
        self.sequence_len = 10
        self.batch_size = 32
        self.input_layer = layers.InputLayer(input_shape=(self.sequence_len, self.state_vars), name="mlp_input_layer")

        # State Embedding
        self.state_embedding_layer = layers.Embedding(
            input_dim=self.sequence_len,
            output_dim=self.state_vars,
            weights=[self.get_pos_encoding_matrix(self.sequence_len, self.state_vars)],
            name="state_sequence_embedding",
        )

        # Action Embedding
        self.total_actions = 5
        self.action_embedding_layer = TokenAndPositionEmbedding(
            config.vocab_size,
            self.sequence_len,
            self.state_vars,
            mask_zero=True
        )

        # Decoders
        self.normalize_first = False
        self.dense_dim = 128
        self.decoder_1 = TransformerDecoder(
            self.dense_dim,
            self.state_vars,
            normalize_first=self.normalize_first,
            name='design_decoder_1'
        )

        # Output Layers
        # - probability distribution over possible actions
        self.output_layer = layers.Dense(self.total_actions, name="action_output_layer")
        self.activation = layers.Activation('linear', dtype='float32')

        # Optimizer
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Hyperparameters
        self.epsilon = 0.99
        self.step = 1

        # Replay Buffer
        self.buffer_size = 10000
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.memory = []
        self.state_memory = []
        self.action_memory = []

    def call(self, inputs, training=True, mask=None):
        states, actions = inputs

        # state_position_embeddings = self.state_embedding_layer(tf.range(start=0, limit=config.state_sequence_len, delta=1))
        state_embeddings = states  # + state_position_embeddings
        action_embeddings = self.action_embedding_layer(actions)
        decoded_actions = self.decoder_1(action_embeddings, encoder_sequence=state_embeddings, training=training, use_causal_mask=True)
        activations = self.output_layer(decoded_actions)
        q_values = self.activation(activations)

        return q_values

    def implicit_build(self):
        state_input = tf.zeros((1, config.state_sequence_len, self.state_vars))
        action_input = tf.zeros((1, config.sequence_len))
        self.call([state_input, action_input])
        return self

    # ------------------------------------
    # Epsilon Greedy
    # ------------------------------------

    def linear_decay(self):
        epsilon_end = 0.05
        total_steps = 600
        if self.step > total_steps:
            return epsilon_end
        return self.epsilon - self.step * (self.epsilon - epsilon_end) / total_steps

    def cosine_decay(self):
        decay_steps = 400
        decay_min = 0.15
        step = min(self.step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - decay_min) * cosine_decay + decay_min
        return self.epsilon * decayed

    # ------------------------------------
    # Input Tensors
    # ------------------------------------

    def get_state_tensor(self, state):
        # print('--> TRACING get_state_tensor')
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)  # State is list of state variables
        state_tensor = tf.expand_dims(state_tensor, axis=0)           # Add batch dimension
        return state_tensor

    def get_state_sequence(self, states):
        # print('--> TRACING get_state_sequence')
        # Pad state sequence
        state_mask = [1] * len(states)
        while len(states) < config.state_sequence_len:
            states.append([0 for _ in range(self.state_vars)])
            state_mask.append(0)
        # state_tensors = [self.get_state_tensor(state) for state in states]
        # state_sequence = tf.concat(state_tensors, axis=0)        #  shape = (sequence_len, state_values:4)


        state_sequence = tf.convert_to_tensor(states, dtype=tf.float32)
        state_sequence = tf.expand_dims(state_sequence, axis=0)  # shape = (1, sequence_len, state_values:4)
        return state_sequence

    def get_state_sequence_batch(self, batch_states):
        # print('--> TRACING get_state_sequence_batch')
        state_tensors = [self.get_state_sequence(states) for states in batch_states]
        state_sequence_batch = tf.concat(state_tensors, axis=0)  #  shape = (batch_size, sequence_len, state_values:4)
        return state_sequence_batch



    # ------------------------------------
    # Actions
    # ------------------------------------

    def get_aciton(self, state, training=True, num_actions=None, debug=False, rand_action=False, init_action=False):


        # State tensor
        if len(self.state_memory) < config.state_sequence_len - 1:
            input_states = deepcopy(self.state_memory)
            input_states.append(state)
        else:
            input_states = deepcopy(self.state_memory[-(config.state_sequence_len - 1):])
        input_states.append(state)
        state_tensor = self.get_state_sequence(input_states)  # shape: (1, sequence_len, state_values)

        # Action tensor
        if len(self.action_memory) < self.sequence_len - 1:
            input_actions = deepcopy(self.action_memory)
        else:
            input_actions = deepcopy(self.action_memory[-(self.sequence_len - 1):])
        input_actions = [str(action) for action in input_actions]
        input_actions.insert(0, '[start]')
        input_actions_str = ' '.join(input_actions)
        action_tensor = config.encode(input_actions_str)
        action_tensor = tf.expand_dims(action_tensor, axis=0)  # shape: (1, sequence_len)
        inference_idx = len(input_actions)-1
        # print('--> ACTION TENSOR:', action_tensor)

        # Forward pass
        q_values = self.call([state_tensor, action_tensor], training=training)
        q_values = q_values[0, inference_idx, :]
        if num_actions:
            q_values = q_values[:num_actions]

        # Select Action
        epsilon = self.linear_decay()
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
            print('STEP', self.step, 'ACTION', action_idx.numpy(), 'EPSILON', round(epsilon, 2), 'Q-VALUES', q_list,
                  'STATE', s_list)

        self.step += 1
        return action_idx.numpy()

    def get_q_value(self, inputs, action_idx=None, training=True):
        # print('--> TRACING get_q_value_batch')
        states, actions = inputs

        # State tensor
        state_tensor = self.get_state_sequence(states)  # shape: (1, sequence_len, state_values)

        # Action tensor
        input_actions = [str(action) for action in actions]
        input_actions = input_actions[:-1]
        input_actions.insert(0, '[start]')
        input_actions = ' '.join(input_actions)
        action_tensor = config.encode(input_actions)

        # Forward pass
        q_values = self.call([state_tensor, action_tensor], training=training)
        q_values = q_values[0, :, :]
        if action_idx:
            q_value = tf.gather(q_values, action_idx)
        else:
            max_idx = tf.argmax(q_values, axis=-1)
            q_value = tf.gather(q_values, max_idx)
        return q_value

    @tf.function
    def get_q_value_idx_batch_fast(self, state_tensors, action_tensors, action_mask_tensors, action_idxs_tensors):
        print('--> TRACING get_q_value_idx_batch_fast')
        batch_size = 32

        # Forward pass
        q_values = self.call([state_tensors, action_tensors], training=True)

        batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
        seq_indices = tf.range(config.sequence_len)[tf.newaxis, :, tf.newaxis]
        combined_indices = tf.concat([
            batch_indices * tf.ones_like(action_idxs_tensors)[:, :, tf.newaxis],
            seq_indices * tf.ones_like(action_idxs_tensors)[:, :, tf.newaxis],
            action_idxs_tensors[:, :, tf.newaxis]
        ], axis=-1)
        q_values = tf.gather_nd(q_values, combined_indices)

        # Mask out Q values for padding actions
        q_values = q_values * action_mask_tensors

        return q_values

    @tf.function
    def get_q_value_max_batch_fast(self, state_tensors, action_tensors, action_mask_tensors):
        print('--> TRACING get_q_value_max_batch_fast')

        # Forward pass
        q_values = self.call([state_tensors, action_tensors], training=True)


        q_values = tf.reduce_max(q_values, axis=-1)

        # Mask out Q values for padding actions
        q_values = q_values * action_mask_tensors

        return q_values



    def get_q_value_batch(self, inputs, action_idxs=None, training=True):
        # print('--> TRACING get_q_value_batch')
        states, actions = inputs
        batch_size = len(states)

        # State tensor
        state_tensors = self.get_state_sequence_batch(states)  # shape: (batch_size, sequence_len, 4)

        # Action tensor
        action_tensors = []
        action_mask_tensors = []
        action_idxs_tensors = []
        for idx, batch_element in enumerate(actions):
            input_actions = [str(action) for action in batch_element]
            input_actions.insert(0, '[start]')
            input_actions_str = ' '.join(input_actions)
            action_tensor = config.encode(input_actions_str)
            action_tensor = tf.expand_dims(action_tensor, axis=0)  # shape: (1, sequence_len)
            action_tensors.append(action_tensor)

            sequence_mask = [1] * len(input_actions)
            while len(sequence_mask) < config.sequence_len:
                sequence_mask.append(0)
            sequence_mask = tf.convert_to_tensor(sequence_mask, dtype=tf.float32)
            sequence_mask = tf.expand_dims(sequence_mask, axis=0)
            action_mask_tensors.append(sequence_mask)

            # Record and Pad action indices
            if action_idxs:
                while len(action_idxs[idx]) < config.sequence_len:
                    action_idxs[idx].append(0)
                action_idxs[idx] = tf.convert_to_tensor(action_idxs[idx], dtype=tf.int64)
                action_idxs[idx] = tf.expand_dims(action_idxs[idx], axis=0)
                action_idxs_tensors.append(action_idxs[idx])




        action_mask_tensors = tf.cast(tf.concat(action_tensors, axis=0), tf.float32)
        action_tensors = tf.concat(action_tensors, axis=0)  # shape: (batch_size, sequence_len)
        if action_idxs:
            action_idxs_tensors = tf.concat(action_idxs_tensors, axis=0)  # shape: (batch_size, sequence_len)
            action_idxs_tensors = tf.cast(action_idxs_tensors, tf.int32)




        # Forward pass
        q_values = self.call([state_tensors, action_tensors], training=training)

        if action_idxs is not None:
            # If action indices are provided, gather Q values based on these indices
            # q_values = tf.gather(q_values, action_idxs, batch_dims=1)

            batch_indices = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
            seq_indices = tf.range(config.sequence_len)[tf.newaxis, :, tf.newaxis]
            combined_indices = tf.concat([
                batch_indices * tf.ones_like(action_idxs_tensors)[:, :, tf.newaxis],
                seq_indices * tf.ones_like(action_idxs_tensors)[:, :, tf.newaxis],
                action_idxs_tensors[:, :, tf.newaxis]
            ], axis=-1)
            q_values = tf.gather_nd(q_values, combined_indices)
        else:
            # If not, take max Q value for each state in the batch
            q_values = tf.reduce_max(q_values, axis=-1)

        # Mask out Q values for padding actions
        q_values = q_values * action_mask_tensors

        return q_values

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(32, 10, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(32, 10), dtype=tf.int32),
        tf.TensorSpec(shape=(32,), dtype=tf.float32),
        tf.TensorSpec(shape=(32,), dtype=tf.float32),
        tf.TensorSpec(shape=(32, 10), dtype=tf.int32),
        tf.TensorSpec(shape=(32, 10), dtype=tf.float32),
    ])
    def train_step(self, batch_states, batch_actions, target_q_value, current_q_values, batch_action_indices, batch_action_mask):  # batch_action_indices replaces action_idxs_tensors
        print('--> TRACING TRAIN')
        # batch_states, batch_actions, target_q_value, current_q_values = inputs
        # current_q_values: (batch_size,)
        # target_q_value: (batch_size,)

        with tf.GradientTape() as tape:
            q_values = self.call([batch_states, batch_actions], training=True)  # shape = (batch_size, seq_length, num_actions)

            # Gather q values for selected actions
            batch_indices = tf.range(self.batch_size)[:, tf.newaxis, tf.newaxis]
            seq_indices = tf.range(config.sequence_len)[tf.newaxis, :, tf.newaxis]
            combined_indices = tf.concat([
                batch_indices * tf.ones_like(batch_action_indices)[:, :, tf.newaxis],
                seq_indices * tf.ones_like(batch_action_indices)[:, :, tf.newaxis],
                batch_action_indices[:, :, tf.newaxis]
            ], axis=-1)
            q_values = tf.gather_nd(q_values, combined_indices)  # shape = (batch_size, seq_length)

            # Mask out Q values for padding actions
            q_values = q_values * batch_action_mask  # shape = (batch_size, seq_length)

            # Sum Q values across sequence length
            q_values = tf.reduce_sum(q_values, axis=-1)  # shape = (batch_size,)

            # Cumulative q values
            q_sum = q_values + current_q_values

            # Loss
            loss = self.loss_fn(target_q_value, q_sum)

        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    # -------------------------------------------
    # Set weights
    # -------------------------------------------

    def load_target_weights(self, model_instance, trainable=False):
        """
        Updates the weights of the current instance with the weights of the provided model instance.
        Args:
            model_instance (SatTransformer): Instance of SatTransformer whose weights will be used.
        """
        for target_layer, source_layer in zip(self.layers, model_instance.layers):
            target_layer.set_weights(source_layer.get_weights())
        self.trainable = trainable

    # -------------------------------------------
    # Experience Replay
    # -------------------------------------------

    def record_experience(self, state, action, reward, next_state, done=False):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.replay_buffer.append(experience)

    def sample_buffer(self, batch_size, indices=None):
        if indices is None:
            indices = random.sample(range(len(self.replay_buffer)), batch_size)
        return [self.replay_buffer[i] for i in indices]

    def sample_memory_trajectory(self, clip_indices):
        clips = []
        for clip_idx in clip_indices:
            clip = deepcopy(self.memory[clip_idx[0]:clip_idx[1]])
            clips.append(clip)
        return clips  # shape = (batch_size, sequence_len, state_values:4)

    # a function that slices a window of memory within a time window start and end time
    def sample_memory_window(self, start_time, end_time):
        window_samples = []
        for experience in self.memory:
            action_time = experience.state[0]  # Assuming the first state variable is the time
            if start_time <= action_time <= end_time:
                window_samples.append(experience)
        return window_samples

    def get_memory_bounds(self):
        if len(self.memory) > 0:
            start_time = self.memory[0].state[0]
            end_time = self.memory[-1].state[0]
            return start_time, end_time
        else:
            return None, None


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

    def get_pos_encoding_matrix(self, max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc









    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

























