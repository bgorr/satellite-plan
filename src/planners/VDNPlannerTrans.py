from planners.AbstractPlanner import AbstractPlanner
from copy import deepcopy
from planners import utils
from models.SatelliteDecoder import SatelliteDecoder
import random
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import config

from planners.TransformerRL import TransformerRL



class VDNPlannerTrans(AbstractPlanner):

    def __init__(self, settings):
        super().__init__(settings)
        self.models_update_frequency = 2
        self.target_update_frequency = 10
        self.buffer_init_size = 64
        self.replay_batch_size = 32
        self.gamma = 0.9
        self.executor = ProcessPoolExecutor(max_workers=12)

    def init_models(self):
        for satellite in self.satellites:
            satellite['q_network'] = SatelliteDecoder().implicit_build()
            satellite['target_q_network'] = SatelliteDecoder().implicit_build()

    def get_satellite_state(self, sat):
        return [
            sat['sat_time'],
            sat['sat_angle'],
            sat['storage'],
            # sat['sat_lat'],
            # sat['sat_lon'],
        ]

    # ---------------------------------------------
    # VDN action
    # ---------------------------------------------

    def select_action(self, sat, state, num_actions, rand_action=False):
        if rand_action is True:
            return random.randint(0, num_actions - 1)
        else:
            return sat['q_network'].get_action(state, sat['experience_buffer'], num_actions=num_actions, debug=False, rand_action=False)



    def update_satellite_models(self):

        trainable_sats = self.satellites

        # --------------------------------
        # 1. Experience buffer sampling
        # --------------------------------

        sat_experience_sequences, sat_prev_actions = self.sample_satellite_time_windows(trainable_sats)

        # --------------------------------
        # 2. Cumulative reward
        # --------------------------------

        # Initialize lists: shapes (num_sats, batch_size, seq_length)
        all_rewards, all_states, all_actions, all_next_states, all_next_actions = [], [], [], [], []

        # Loop through satellite experience sequences
        for experience_sequence in sat_experience_sequences:
            sat_sequence_states = VDNPlannerTrans.extract_idx_from_sequences(experience_sequence, 0)
            sat_sequence_actions = VDNPlannerTrans.extract_idx_from_sequences(experience_sequence, 1)
            sat_sequence_rewards = VDNPlannerTrans.extract_idx_from_sequences(experience_sequence, 2)
            sat_sequence_next_states = VDNPlannerTrans.extract_idx_from_sequences(experience_sequence, 3)

            # Pad rewards
            for sequence_rewards, sequence_actions in zip(sat_sequence_rewards, sat_sequence_actions):
                while len(sequence_rewards) < config.sequence_len:
                    sequence_rewards.append(0.0)
                while len(sequence_actions) < config.sequence_len:
                    sequence_actions.append(0)

            # Remove first action from next actions
            sat_sequence_next_actions = [a[1:] for a in sat_sequence_actions]

            all_rewards.append(sat_sequence_rewards)
            all_states.append(sat_sequence_states)
            all_actions.append(sat_sequence_actions)
            all_next_states.append(sat_sequence_next_states)
            all_next_actions.append(sat_sequence_next_actions)

        # Convert to tensors and compute cumulative rewards
        cum_rewards = tf.convert_to_tensor(all_rewards)
        cumulative_reward_seqwise = tf.reduce_sum(cum_rewards, axis=0)  # (batch_size, seq_length)
        cumulative_reward = tf.reduce_sum(cum_rewards, axis=-1)  # (num_sats, batch_size)
        cumulative_reward = tf.reduce_sum(cumulative_reward, axis=0)  # (batch_size)
        cumulative_reward = tf.cast(cumulative_reward, dtype=tf.float32)

        # --------------------------------
        # Next Actions
        # --------------------------------

        sat_data_list = []
        for idx1, sat in enumerate(trainable_sats):
            sat_data = {
                'all_states': all_states[idx1],
                'all_next_states': all_next_states[idx1],
                'all_actions': all_actions[idx1],
                'all_next_actions': all_next_actions[idx1],
                'state_vars': sat['q_network'].state_vars,
                'prev_actions': sat_prev_actions[idx1]
            }
            sat_data_list.append(sat_data)

        # Initialize result lists
        batch_state_sequences, batch_next_state_sequences = [], []
        batch_action_tensors, batch_action_mask, batch_action_indices = [], [], []
        batch_next_action_tensors, batch_next_action_mask = [], []
        batch_state_mask, batch_next_state_mask = [], []

        # Execute in parallel
        results = list(self.executor.map(VDNPlannerTrans.process_actions_proc, sat_data_list))

        for result in results:
            batch_state_sequences.append(result['batch_state_sequences'])
            batch_next_state_sequences.append(result['batch_next_state_sequences'])
            batch_action_tensors.append(result['batch_action_tensors'])
            batch_action_mask.append(result['batch_action_mask'])
            batch_action_indices.append(result['batch_action_indices'])
            batch_next_action_tensors.append(result['batch_next_action_tensors'])
            batch_next_action_mask.append(result['batch_next_action_mask'])
            batch_state_mask.append(result['batch_state_masks'])
            batch_next_state_mask.append(result['batch_next_state_masks'])

        # --------------------------------
        # Q Values
        # --------------------------------

        # 3.3. (target_q_network) Find summed q_values for "next" state
        target_q_values = []  # shape: (num_sats, batch_size, seq_length)
        for idx, sat in enumerate(trainable_sats):
            target_q_value = sat['target_q_network'].get_q_value_max_batch_fast(
                batch_next_state_sequences[idx],
                batch_next_action_tensors[idx],
                batch_next_action_mask[idx],
                batch_next_state_mask[idx],
            )
            target_q_values.append(target_q_value)  # shape: (batch_size, seq_length)
        summed_q_value_next_seqwise = tf.stack(target_q_values, axis=0)  # shape: (num_sats, batch_size, seq_length)
        summed_q_value_next = tf.reduce_sum(target_q_values, axis=-1)  # shape: (num_sats, batch_size)  sum sat timestep contribution
        summed_q_value_next = tf.reduce_sum(summed_q_value_next, axis=0)  # shape: (batch_size,)  sum sat contribution
        summed_q_value_next = tf.cast(summed_q_value_next, dtype=tf.float32)

        q_targets = cumulative_reward + self.gamma * summed_q_value_next  # shape: (batch_size,)
        q_targets = tf.cast(q_targets, dtype=tf.float32)
        q_targets_seqwise = cumulative_reward_seqwise + self.gamma * summed_q_value_next_seqwise  # shape: (batch_size, seq_length)
        q_targets_seqwise = tf.cast(q_targets_seqwise, dtype=tf.float32)
        q_targets_inputs = [q_targets] * len(trainable_sats)
        q_targets_inputs_seqwise = [q_targets_seqwise] * len(trainable_sats)

        # 3.2. (q_network) Find summed q_values for last (actually current) state
        last_q_values = []  # shape: (num_sats, batch_size, seq_length)
        last_q_values_sum = []
        last_q_values_sum_seqwise = []
        for idx, sat in enumerate(trainable_sats):
            last_q_value = sat['q_network'].get_q_value_idx_batch_fast(
                batch_state_sequences[idx],
                batch_action_tensors[idx],
                batch_action_mask[idx],
                batch_action_indices[idx],
                batch_state_mask[idx]
            )
            last_q_values.append(last_q_value)  # shape: (batch_size, seq_length)
            last_q_value_sum = tf.reduce_sum(last_q_value, axis=-1)  # shape: (batch_size,)

            last_q_values_sum_seqwise.append(last_q_value)  # shape: (batch_size, seq_length)
            last_q_values_sum.append(last_q_value_sum)  # shape: (batch_size,)
        last_q_tensor = tf.stack(last_q_values)  # shape: (num_sats, batch_size, seq_length)
        total_sum_seqwise = tf.reduce_sum(last_q_tensor, axis=0)  # shape: (batch_size, seq_length)
        total_sum = tf.reduce_sum(last_q_tensor, axis=-1)  # shape: (num_sats, batch_size)
        total_sum = tf.reduce_sum(total_sum, axis=0)  # shape: (batch_size,)
        external_sat_sum = []  # shape: (num_sats, batch_size)
        external_sat_sum_seqwise = []  # shape: (num_sats, batch_size, seq_length)
        for idx in range(len(trainable_sats)):
            # Subtract the Q-value of the current satellite from the total sum
            sum_without_current_sat = total_sum - last_q_values_sum[idx]  # shape: (batch_size, )
            external_sat_sum.append(
                tf.cast(sum_without_current_sat, dtype=tf.float32)
            )
            external_sat_sum_seqwise.append(
                tf.cast(total_sum_seqwise - last_q_values_sum_seqwise[idx], dtype=tf.float32)
            )

        # Create Dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            batch_state_sequences,
            batch_action_tensors,
            q_targets_inputs,
            external_sat_sum,
            batch_action_indices,
            batch_action_mask
        ))
        dataset_seqwise = tf.data.Dataset.from_tensor_slices((
            batch_state_sequences,
            batch_action_tensors,
            q_targets_inputs_seqwise,
            external_sat_sum_seqwise,
            batch_action_indices,
            batch_action_mask
        ))

        # Train
        losses = []
        # for idx, data_tuple in enumerate(dataset):
        #     batch_states, batch_actions, target_q_value, current_q_values, batch_action_indices, batch_action_mask = data_tuple
        #     metrics = trainable_sats[idx]['q_network'].train_step(batch_states, batch_actions, target_q_value,
        #                                                           current_q_values, batch_action_indices,
        #                                                           batch_action_mask)
        #     losses.append(metrics['loss'])

        for idx, data_tuple in enumerate(dataset_seqwise):
            batch_states_seqwise, batch_actions_seqwise, target_q_value_seqwise, current_q_values_seqwise, batch_action_indices_seqwise, batch_action_mask_seqwise = data_tuple
            metrics_seqwise = trainable_sats[idx]['q_network'].train_step_seqwise(batch_states_seqwise, batch_actions_seqwise,
                                                                           target_q_value_seqwise,
                                                                           current_q_values_seqwise,
                                                                           batch_action_indices_seqwise,
                                                                           batch_action_mask_seqwise)
            losses.append(metrics_seqwise['loss'])
        print('--> AVERAGE LOSS:', np.mean(losses), 'TIME', self.satellites[0]['sat_time'])




    def sample_experiences(self, trainable_sats):
        min_reel_len = np.min([len(sat['experience_buffer']) for sat in trainable_sats])
        num_reels = min([len(sat['experience_reels']) for sat in trainable_sats])
        if num_reels == 0:
            return self.sample_sync_experiences(trainable_sats)
        else:
            prob_cur_reel = (1.0 / (num_reels + 1))
            if random.random() < prob_cur_reel and min_reel_len > self.replay_batch_size:
                return self.sample_sync_experiences(trainable_sats)
            else:
                # print('--> SAMPLING REEL')
                return self.sample_sync_reel_experiences(trainable_sats)
        # return self.sample_sync_experiences(trainable_sats)

    def sample_sync_experiences(self, trainable_sats, min_buf_size=1e9):

        for sat in trainable_sats:
            if len(sat['experience_buffer']) < min_buf_size:
                min_buf_size = len(sat['experience_buffer'])

        sat_experiences = []  # (num_sats, batch_size, 5)
        indices = random.sample(range(int(min_buf_size)), self.replay_batch_size)
        for sat in trainable_sats:
            sat_experiences.append([sat['experience_buffer'][i] for i in indices])
        return sat_experiences

    def sample_sync_reel_experiences(self, trainable_sats):
        num_reels = min([len(sat['experience_reels']) for sat in trainable_sats])
        if num_reels < 1:
            return None

        # rand_reel_idx = random.randint(0, num_reels - 1)
        rand_reel_batch = [random.randint(0, num_reels - 1) for _ in range(self.replay_batch_size)]
        # rand_reel_batch = [(num_reels - 1) for _ in range(self.replay_batch_size)]

        min_buf_size = 1e9
        for sat in trainable_sats:
            for exp_reel in sat['experience_reels']:
                if len(exp_reel) < min_buf_size:
                    min_buf_size = len(exp_reel)

        sat_experiences = []  # (num_sats, batch_size, 5)
        indices = random.sample(range(int(min_buf_size)), self.replay_batch_size)
        for sat in trainable_sats:
            # sat_experiences.append([sat['experience_reels'][rand_reel_idx][i] for i in indices])
            sat_experiences.append([sat['experience_reels'][r][i] for r, i in zip(rand_reel_batch, indices)])
        return sat_experiences






    def sample_satellite_time_windows(self, trainable_sats):

        memory_matrix = []
        for sat in trainable_sats:
            memory_row = [exp[0][0] for exp in sat['experience_buffer']]
            memory_matrix.append(memory_row)

        reel_matrices = [memory_matrix]
        for idx, reel in enumerate(self.satellites[0]['experience_reels']):
            reel_matrix = []
            for sat in trainable_sats:
                memory_row = [exp[0][0] for exp in sat['experience_reels'][idx]]
                reel_matrix.append(memory_row)
            reel_matrices.append(reel_matrix)


        sat_clips = [[]]
        sat_prev_actions = []
        # for batch_element_idx in range(self.replay_batch_size):
        while len(sat_clips[0]) < self.replay_batch_size:
            reel_matrix_idx = random.randint(0, len(reel_matrices) - 1)
            reel_matrix = reel_matrices[reel_matrix_idx]

            # Check if the reel matrix is long enough
            min_len = min([len(reel_row) for reel_row in reel_matrix])
            if min_len < 10:
                continue

            # Select random length for batch element time window between 2 and sequence_len
            action_ub = min(min_len, config.sequence_len)
            action_lb = max(2, action_ub - 5)
            action_len = random.randint(action_lb, action_ub)
            action_start_idx = random.randint(0, min_len - action_len)
            for sat_idx, sat in enumerate(trainable_sats):
                if len(sat_clips) < sat_idx+1:
                    sat_clips.append([])
                if len(sat_prev_actions) < sat_idx+1:
                    sat_prev_actions.append([])

                if reel_matrix_idx == 0:
                    memory_reel = sat['experience_buffer']
                else:
                    memory_reel = sat['experience_reels'][reel_matrix_idx-1]

                sat_clips[sat_idx].append(
                    deepcopy(memory_reel[action_start_idx:action_start_idx+action_len])
                )
                prev_action_idx = action_start_idx-1
                if prev_action_idx < 0:
                    sat_prev_actions[sat_idx].append(
                        '[start]'
                    )
                else:
                    sat_prev_actions[sat_idx].append(
                        deepcopy(str(memory_reel[prev_action_idx][1]))
                    )

        return sat_clips, sat_prev_actions

    def sample_satellite_time_windows_staggered(self, trainable_sats):

        def sample_time_window(ragged_memory_matrix, lower_bound_max, upper_bound_min):
            t_start = random.uniform(lower_bound_max, upper_bound_min)
            end_times = []
            for sat_memory_seq in ragged_memory_matrix:
                sat_end_times = []
                for memory_time in sat_memory_seq:
                    if t_start > memory_time:
                        continue
                    sat_end_times.append(memory_time)
                    if len(sat_end_times) >= config.sequence_len:
                        break
                if len(sat_end_times) < config.sequence_len:
                    return None
                end_times.append(sat_end_times)

            # assumptions, 10 experience replay steps exist for each sat
            # find idx of sat whose end time is smallest, this will be the sat that keeps all of its experience
            min_idx = np.argmin([x[-1] for x in end_times])
            min_end_time = end_times[min_idx][-1]
            pruned_end_times = []
            for idx, action_times in enumerate(end_times):
                if idx == min_idx:
                    pruned_end_times.append(action_times)
                else:
                    pruned_items = [time_e for time_e in action_times if time_e <= min_end_time]
                    if len(pruned_items) == 0:
                        return None
                    pruned_end_times.append(pruned_items)
            return pruned_end_times

        def sample_time_window_batch(ragged_memory_matrix, batch_size):
            upper_bound_min = min([x[-1] for x in ragged_memory_matrix])
            lower_bound_max = max([x[0] for x in ragged_memory_matrix])

            # 1. Pick random start time
            # 2. Find all end times for each satellite
            batch = []
            while (len(batch) < batch_size):
                window = sample_time_window(ragged_memory_matrix, lower_bound_max, upper_bound_min)
                counter = 0
                while window is None:
                    window = sample_time_window(ragged_memory_matrix, lower_bound_max, upper_bound_min)
                    counter += 1
                    if counter > 100:
                        raise ValueError("--> COULDNT SAMPLE TIME WINDOW")
                batch.append(window)
            return batch

        memory_matrix = []
        for sat in trainable_sats:
            memory_row = [exp[0][0] for exp in sat['experience_buffer']]
            memory_matrix.append(memory_row)

        reel_matrices = [memory_matrix]
        for idx, reel in self.satellites[0]['experience_reels']:
            reel_matrix = []
            for sat in trainable_sats:
                memory_row = [exp[0][0] for exp in sat['experience_reels'][idx]]
                reel_matrix.append(memory_row)
            reel_matrices.append(reel_matrix)



        memory_batch = sample_time_window_batch(memory_matrix, self.replay_batch_size)
        clip_indices = []
        for batch in memory_batch:  # Iterate over each batch element
            for idx, sat_memory in enumerate(batch):  # Iterate over each batch element satellite
                if len(clip_indices) <= idx:
                    clip_indices.append([])
                action_start_idx = memory_matrix[idx].index(sat_memory[0])
                action_end_idx = memory_matrix[idx].index(sat_memory[-1])
                clip_indices[idx].append([action_start_idx, action_end_idx])

        sat_experience_sequences = []  # (num_sats, batch_size, seq_length, 5)
        for idx, sat in enumerate(trainable_sats):
            clip_index = clip_indices[idx]
            clips = []
            for clip_idx in clip_index:
                clip = deepcopy(sat['experience_buffer'][clip_idx[0]:clip_idx[1]])
                clips.append(clip)
            sat_experience_sequences.append(clips)

        return sat_experience_sequences

    @staticmethod
    def extract_idx_from_sequences(experience_sequences, key):
        return [utils.idx_from_batch(sequence_memory, key) for sequence_memory in experience_sequences]

    @staticmethod
    def process_actions_proc(sat_data):

        batch_states = sat_data['all_states']
        batch_next_states = sat_data['all_next_states']
        batch_actions = sat_data['all_actions']
        batch_next_actions = sat_data['all_next_actions']
        state_vars = sat_data['state_vars']
        prev_actions = sat_data['prev_actions']

        # Padding states
        batch_states_mask = []
        for states in batch_states:
            mask = [1] * len(states)
            while len(states) < config.sequence_len:
                states.append([0] * state_vars)
                mask.append(0)
            batch_states_mask.append(mask)

        # Padding next states
        batch_next_states_mask = []
        for states in batch_next_states:
            mask = [1] * len(states)
            while len(states) < config.sequence_len:
                states.append([0] * state_vars)
                mask.append(0)
            batch_next_states_mask.append(mask)

        # Convert to tensors
        state_sequence_tensor = tf.convert_to_tensor(batch_states, dtype=tf.float32)
        next_state_sequence_tensor = tf.convert_to_tensor(batch_next_states, dtype=tf.float32)

        state_mask_tensor = tf.convert_to_tensor(batch_states_mask, dtype=tf.float32)
        next_state_mask_tensor = tf.convert_to_tensor(batch_next_states_mask, dtype=tf.float32)

        # Process actions
        action_tensors, action_mask_tensors, action_idxs_tensors = VDNPlannerTrans.process_batch_actions(batch_actions, prev_actions)
        next_action_tensors, next_action_mask_tensors, next_action_idxs_tensors = TransformerRL.process_batch_actions(
            batch_next_actions)

        return {
            'batch_state_sequences': state_sequence_tensor,
            'batch_next_state_sequences': next_state_sequence_tensor,
            'batch_action_tensors': action_tensors,
            'batch_action_mask': action_mask_tensors,
            'batch_action_indices': action_idxs_tensors,
            'batch_next_action_tensors': next_action_tensors,
            'batch_next_action_mask': next_action_mask_tensors,
            'batch_state_masks': state_mask_tensor,
            'batch_next_state_masks': next_state_mask_tensor
        }

    @staticmethod
    def process_batch_actions(batch_actions, prev_actions):
        action_tensors, action_mask_tensors, action_idxs_tensors = [], [], []
        for idx, batch_element in enumerate(batch_actions):
            prev_action = prev_actions[idx]
            # input_actions = ['[start]'] + [str(action) for action in batch_element]
            input_actions = [prev_action] + [str(action) for action in batch_element]
            if len(input_actions) > config.sequence_len:
                input_actions = input_actions[:config.sequence_len]
            input_actions_str = ' '.join(input_actions)

            action_tensor = config.encode(input_actions_str)
            action_tensors.append(action_tensor)

            sequence_mask = [1] * len(input_actions) + [0] * (config.sequence_len - len(input_actions))
            action_mask_tensors.append(tf.convert_to_tensor(sequence_mask, dtype=tf.float32))

            padded_actions = batch_element + [0] * (config.sequence_len - len(batch_element))
            action_idxs_tensors.append(tf.convert_to_tensor(padded_actions, dtype=tf.int64))

        return tf.convert_to_tensor(action_tensors, dtype=tf.int32), \
            tf.convert_to_tensor(action_mask_tensors, dtype=tf.float32), \
            tf.convert_to_tensor(action_idxs_tensors, dtype=tf.int32)







