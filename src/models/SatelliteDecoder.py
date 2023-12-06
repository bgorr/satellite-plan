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
import tensorflow_addons as tfa


import config

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@keras.saving.register_keras_serializable(package="PlanningAndScheduling", name="SatelliteDecoder")
class SatelliteDecoder(tf.keras.Model):

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_tracker = tf.keras.metrics.Mean(name="loss")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1

        # ----- VARIABLES -----
        self.total_actions = config.num_actions
        self.action_embed_dim = 32
        self.sequence_len = config.sequence_len
        self.batch_size = 32
        self.dense_dim = 128

        # ----- OPTIMIZER -----
        self.learning_rate = 0.0001
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=self.learning_rate)

        # ----- POLICY -----
        self.epsilon = 0.99
        self.epsilon_end = 0.01
        self.decay_steps = 1000

        # Replay Buffer
        self.buffer_size = 10000
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.memory = []
        self.state_memory = []
        self.action_memory = []

        # ----- STATE SEQUENCE -----
        # 1. Satellite mission_time
        # 2. Satellite slewing angle
        # 3. Satellite storage
        # 4. Satellite latitude
        # 5. Satellite longitude
        self.state_vars = 3
        # self.state_embedding_layer = layers.Embedding(
        #     input_dim=self.sequence_len,
        #     output_dim=self.state_vars,
        #     weights=[self.get_pos_encoding_matrix(self.sequence_len, self.state_vars)],
        #     name="state_sequence_embedding",
        # )
        self.state_embedding_layer = layers.Dense(self.action_embed_dim, name="state_dim_expansion")

        # ----- ACTION SEQUENCE -----
        self.action_embedding_layer = TokenAndPositionEmbedding(
            config.vocab_size,
            self.sequence_len,
            self.action_embed_dim,
            mask_zero=True
        )

        # ----- DECODER -----
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(
            self.dense_dim,
            self.action_embed_dim,
            normalize_first=self.normalize_first,
            name='action_decoder_1'
        )

        # ----- OUTPUT LAYER -----
        self.output_layer = layers.Dense(self.total_actions, name="action_output_layer")
        self.activation = layers.Activation('linear', dtype='float32')



    def call(self, inputs, training=True, mask=None):
        states, actions = inputs

        # 0. Normalize states
        norm_values = tf.constant([8640.0, 45.0, 1.0], dtype=tf.float32)
        states = states / norm_values

        # 1. State sequence
        # state_embeddings = states
        state_embeddings = self.state_embedding_layer(states)  # Dense embedding layer
        # state_position_embeddings = self.state_embedding_layer(tf.range(start=0, limit=config.sequence_len, delta=1))
        # state_embeddings = states + state_position_embeddings

        # 2. Action sequence
        action_embeddings = self.action_embedding_layer(actions)

        # 3. Decoder
        decoded_actions = self.decoder_1(action_embeddings, encoder_sequence=state_embeddings, training=training, use_causal_mask=True, encoder_padding_mask=mask)

        # 4. Output layer
        activations = self.output_layer(decoded_actions)
        q_values = self.activation(activations)

        return q_values

    def implicit_build(self):
        state_input = tf.zeros((1, self.sequence_len, self.state_vars))
        action_input = tf.zeros((1, self.sequence_len))
        self.call([state_input, action_input])
        return self

    # ------------------------------------
    # Epsilon Greedy
    # ------------------------------------

    def linear_decay(self):
        if self.step > self.decay_steps:
            return self.epsilon_end
        return self.epsilon - self.step * (self.epsilon - self.epsilon_end) / self.decay_steps

    def cosine_decay(self):
        step = min(self.step, self.decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.epsilon_end) * cosine_decay + self.epsilon_end
        return self.epsilon * decayed

    # ------------------------------------
    # Actions
    # ------------------------------------

    def get_action(self, state, trajectory, num_actions=None, debug=False, rand_action=False):
        if num_actions is None:
            num_actions = self.total_actions

        # 1. Get previous states / actions for context
        context_len = config.sequence_len - 1
        if len(trajectory) <= context_len:
            prev_states = [exp[0] for exp in trajectory]
            prev_actions = [str(exp[1]) for exp in trajectory]
            all_actions = ['[start]'] + deepcopy(prev_actions)
        else:
            prev_states = [exp[0] for exp in trajectory[-context_len:]]
            all_actions = [str(exp[1]) for exp in trajectory[-(context_len + 1):]]
        all_states = prev_states + [state]
        next_actions = deepcopy(all_actions[1:])

        # 2. Pad state sequence and get mask
        state_mask = [1] * len(all_states)
        while len(all_states) < self.sequence_len:
            all_states.append([0 for _ in range(self.state_vars)])
            state_mask.append(0)
        state_mask = tf.convert_to_tensor(state_mask)
        state_mask = tf.expand_dims(state_mask, axis=0)
        state_tensor = tf.convert_to_tensor(all_states, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, axis=0)  # shape = (1, sequence_len, 4)

        # 3. Tokenize and pad action sequence
        all_actions_str = ' '.join(all_actions)
        action_tensor = config.encode(all_actions_str)
        action_tensor = tf.expand_dims(action_tensor, axis=0)  # shape: (1, sequence_len)
        inference_idx = len(all_actions) - 1

        # Forward pass
        q_values = self.call([state_tensor, action_tensor], training=False, mask=state_mask)
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

    # ------------------------------------
    # Training
    # ------------------------------------

    @tf.function
    def get_q_value_idx_batch_fast(self, state_tensors, action_tensors, action_mask_tensors, action_idxs_tensors, state_mask):
        print('--> TRACING get_q_value_idx_batch_fast')

        # Forward pass
        q_values = self.call([state_tensors, action_tensors], mask=state_mask)

        batch_indices = tf.range(self.batch_size)[:, tf.newaxis, tf.newaxis]
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
    def get_q_value_max_batch_fast(self, state_tensors, action_tensors, action_mask_tensors, state_mask):
        print('--> TRACING get_q_value_max_batch_fast')

        # Forward pass
        q_values = self.call([state_tensors, action_tensors], training=True, mask=state_mask)

        q_values = tf.reduce_max(q_values, axis=-1)

        # Mask out Q values for padding actions
        q_values = q_values * action_mask_tensors

        return q_values

    @tf.function
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

    @tf.function
    def train_step_seqwise(self, batch_states, batch_actions, target_q_value, current_q_values, batch_action_indices,
                   batch_action_mask):  # batch_action_indices replaces action_idxs_tensors
        print('--> TRACING TRAIN')
        # batch_states, batch_actions, target_q_value, current_q_values = inputs
        # current_q_values: (batch_size,)
        # target_q_value: (batch_size,)

        with tf.GradientTape() as tape:
            q_values = self.call([batch_states, batch_actions],
                                 training=True)  # shape = (batch_size, seq_length, num_actions)

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

            # Cumulative q values
            q_sum = q_values + current_q_values # shape = (batch_size, seq_length)

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
            model_instance (SatelliteDecoder): Instance of SatTransformer whose weights will be used.
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

    def get_memory_bounds(self):
        if len(self.memory) > 0:
            start_time = self.memory[0].state[0]
            end_time = self.memory[-1].state[0]
            return start_time, end_time
        else:
            return None, None

    def get_memory_time_values(self):
        return [experience.state[0] for experience in self.memory]

    # -------------------------------------------
    # Positional Encoding
    # -------------------------------------------

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

























