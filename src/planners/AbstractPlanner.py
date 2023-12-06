import numpy as np
import os
import matplotlib.gridspec as gridspec
import random
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import config
from copy import deepcopy
from actions.Observation import Observation
from actions.Downlink import Downlink

# Utility functions
from planners import utils




class AbstractPlanner:

    def __init__(self, settings):
        self.pool_size = 32  # config.cores
        self.directory = settings["directory"] + "orbit_data/"
        self.settings = settings
        self.satellites = self.init_satellites()
        self.init_models()

        # Episode variables
        self.episode = 0
        self.step = 0

        # Training variables
        self.models_update_frequency = 1
        self.target_update_frequency = 10
        self.buffer_init_size = 100
        self.plot_frequency = 100

        # Plotting variables
        self.num_observations = [0]
        self.num_mem_overflows = [0]
        self.num_events_seen = [0]
        self.num_unique_events_seen = [0]
        self.num_infeasibilities = [0]
        self.num_dl_events = [0]
        self.reward_history = []

        self.training_history = []

    # ---------------------------------------------
    # Initialize
    # ---------------------------------------------

    def init_satellites(self):
        satellites = []
        for key in sorted(self.settings['orbit_data'].keys()):
            sat_orbit_data = self.settings['orbit_data'][key]
            satellites.append({
                "key": key,
                "orbitpy_id": sat_orbit_data['orbitpy_id'],
                "num_actions": 5,

                # --> Experience Buffer
                'experience_buffer': [],  # (state, action, reward, next_state, done)
                'experience_reels': [],    # (reel_num, state, action, reward, next_state, done)
                'critic_reels': [],

                # --> Observation opportunities
                'all_obs': deepcopy(sat_orbit_data['all_obs']),  # All observations seen by the satellite
                'all_last_obs': [],
                'all_gs': deepcopy(sat_orbit_data['gs_time_step_windows']),  # All ground stations seen by the satellite
                'all_comm': deepcopy(sat_orbit_data['comm_windows']),  # All communication windows seen by the satellite
                'planned_obs': [],  # All observations planned by the satellite

                # --> Transition Info
                'last_obs': None,  # The last observation taken by the satellite
                'has_actions': True,  # Records if sat has any actions left to take
                'location_list': [],  # Locations seen by sat
                'last_action_idx': None,  # The index of the last action taken (for delayed back-prop)
                'rewards': [],  # All rewards received by the satellite (for delayed back-prop)
                'took_action': False,
                'nadir_lat_lons': sat_orbit_data['nadir_lat_lons'],  # (time, lat, lon)

                # --> State
                'sat_step': 0,
                'sat_time': 0.0,  # steps
                'sat_angle': 0.0,  # deg
                'sat_lat': sat_orbit_data['nadir_lat_lons'][0][1],
                'sat_lon': sat_orbit_data['nadir_lat_lons'][0][2],
                'last_state': None,  # The input state for the last action taken (for delayed back-prop)
                'sat_obs_idx': 0,  # The index of the last observation taken
                'storage': [],  # holds either events 'E' or non-events 'N'

                # --> Model
                'q_network': None,
                'target_q_network': None,
                'critic': None,
                'critic_values': [],
                'mixing': None,
                'policy': None,


            })
            # print('--> SAT OBS', len(satellites[-1]['all_obs']))
        return satellites

    def init_models(self):
        pass

    # ---------------------------------------------
    # Reset
    # ---------------------------------------------

    def reset_episode(self):

        # 1. Reset step and plotting variables
        self.step = 0
        self.num_observations = [0]
        self.num_mem_overflows = [0]
        self.num_events_seen = [0]
        self.num_unique_events_seen = [0]
        self.num_infeasibilities = [0]
        self.num_dl_events = [0]

        # 2. Reset events
        self.reset_events()

        # 3. Reset satellites
        self.reset_satellites()

    def reset_events(self):
        for grid_point in self.settings['grid_points']:
            grid_point['times_observed'] = 0
            for event in grid_point['events']:
                event['times_observed'] = 0

    def reset_satellites(self):
        for sat in self.satellites:
            sat['sat_time'] = 0.0
            sat['sat_step'] = 0.0
            sat['sat_angle'] = 0.0
            sat['storage'] = []
            sat['last_obs'] = None
            sat['last_action_idx'] = None
            sat['sat_lat'] = sat["nadir_lat_lons"][0][1]
            sat['sat_lon'] = sat["nadir_lat_lons"][0][2]
            sat['has_actions'] = True
            sat['experience_buffer'] = []
            sat['planned_obs'] = []
            sat['rewards'] = []
            sat['sat_obs_idx'] = 0
            sat['critic_values'] = []

    # ---------------------------------------------
    # Record
    # ---------------------------------------------

    def record_step(self):
        self.step += 1
        self.num_observations.append(0)
        self.num_mem_overflows.append(0)
        self.num_events_seen.append(0)
        self.num_unique_events_seen.append(0)
        self.num_infeasibilities.append(0)
        self.num_dl_events.append(0)

    def record_episode(self):
        total_reward = 0
        for sat in self.satellites:
            rewards = [float(experience[2]) for experience in sat['experience_buffer']]
            total_reward += sum(rewards)
        self.reward_history.append(total_reward)
        self.training_history.append(deepcopy(
            self.get_plot_items()
        ))
        for sat in self.satellites:
            if len(sat['experience_buffer']) > 0:
                sat['experience_reels'].append(deepcopy(sat['experience_buffer']))
            if len(sat['critic_values']) > 0:
                sat['critic_reels'].append(deepcopy(sat['critic_values']))

    # ---------------------------------------------
    # Train episode
    # ---------------------------------------------

    def train_episode(self):
        self.reset_episode()

        actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]
        while (len(actionable_sats) > 0):

            # 1. Take a step in environment, run critic
            if self.sim_step(actionable_sats) is False:
                break


            # 2. Update model if enough experiences | reset actions
            min_buffer = min([len(sat['experience_buffer']) for sat in self.satellites])
            if min_buffer > self.buffer_init_size or len(self.satellites[0]['experience_reels']) > 0:
                self.update_satellite_models()
            for sat in self.satellites:
                sat['took_action'] = False

            # 3. Copy of target network (VDNs / Q-Learning)
            if self.step % self.target_update_frequency == 0:
                for sat in self.satellites:
                    if sat['target_q_network']:
                        sat['target_q_network'].load_target_weights(sat['q_network'])

            # 4. Plot progress
            # if self.step > 0 and self.step % self.plot_frequency == 0:
            #     self.plot_progress()

            # 5. Determine actionable satellites
            actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]

            # 6. Increment counters
            if len(actionable_sats) != 0:
                self.record_step()

            # if self.step > 50:
            #     break

        self.record_episode()
        self.plot_history()
        self.episode += 1
        return self.satellites


    # ---------------------------------------------
    # Satellite action
    # ---------------------------------------------

    def satellite_action(self, sat, debug=False):
        # Select an action and record the transition
        # experience = (state, action, reward, next_state, done)

        ################
        ### 1. State ###
        ################

        state = deepcopy(self.get_satellite_state(sat))

        #################
        ### 2. Action ###
        #################

        actions = Observation.get_actions(
            sat,
            self.settings,
            top_n=sat['num_actions']
        )
        # gs_actions = Downlink.get_actions(
        #     sat,
        #     self.settings
        # )

        if len(actions) == 0:  # or len(gs_actions) == 0:
            sat['has_actions'] = False
            sat['took_action'] = False
            return False

        # rand_action = False
        # if len(sat['experience_buffer']) < self.buffer_init_size and len(sat['experience_reels']) == 0:
        #     rand_action = True
        action_log_prob = 0
        # action_idx = self.select_action(sat, state, num_actions=len(actions) + len(gs_actions), rand_action=False)
        action_idx = self.select_action(sat, state, num_actions=len(actions), rand_action=False)
        if type(action_idx) == list:
            action_log_prob = action_idx[1]
            action_idx = action_idx[0]

        #################
        ### 3. Reward ###
        #################

        # IF: gs action
        # if action_idx >= len(actions):
        #     action_obs = gs_actions[0]
        #     reward = action_obs["reward"]
        #     dl_events = action_obs["reward"]
        #     self.num_dl_events[-1] += dl_events
        # else:  # ELSE: obs action
        #     action_obs = actions[action_idx]
        #     reward = action_obs["reward"]
        #     if action_obs['mem-overflow'] is True:
        #         self.num_mem_overflows[-1] += 1
        #     if action_obs['type'] in ['E', 'NE']:
        #         self.num_events_seen[-1] += 1
        #     if action_obs['type'] in ['NE']:
        #         self.num_unique_events_seen[-1] += 1
        #     self.num_observations[-1] += 1
        #     sat['planned_obs'].append(action_obs)


        # IF no gs action possible

        action_obs = actions[action_idx]
        reward = action_obs["reward"]
        if action_obs['mem-overflow'] is True:
            self.num_mem_overflows[-1] += 1
        if action_obs['type'] in ['E', 'NE']:
            self.num_events_seen[-1] += 1
        if action_obs['type'] in ['NE']:
            self.num_unique_events_seen[-1] += 1
        self.num_observations[-1] += 1
        sat['planned_obs'].append(action_obs)

        sat['took_action'] = True

        #####################
        ### 4. Next State ###
        #####################

        sat['sat_time'] = action_obs["soonest"]
        sat['sat_angle'] = action_obs["angle"]
        sat['storage'] = action_obs["storage"]
        sat['sat_step'] = deepcopy(self.step)
        sat_time_approx = int(action_obs["soonest"])
        nadir_lat_lons = sat['nadir_lat_lons'][sat_time_approx]
        time, sat['sat_lat'], sat['sat_lon'] = nadir_lat_lons[0], nadir_lat_lons[1], nadir_lat_lons[2]
        sat['last_obs'] = deepcopy(action_obs)
        sat['sat_obs_idx'] = utils.get_obs_idx(sat['sat_time'], sat['all_obs'])
        if 'storage_img_type' in action_obs:
            sat['storage_img_type'] = action_obs['storage_img_type']
        next_state = deepcopy(self.get_satellite_state(sat))

        ###################
        ### 5. Terminal ###
        ###################

        next_actions, num_infeasible = utils.get_action_space_info(
            sat['sat_time'],
            sat['sat_angle'],
            sat['all_obs'],
            sat['last_obs'],
            self.settings,
            top_n=sat['num_actions']
        )
        self.num_infeasibilities[-1] += num_infeasible
        done = (len(next_actions) == 0)
        if done is True:
            sat['has_actions'] = False
        # infeasible_penalty = -0.1 * num_infeasible
        # reward += infeasible_penalty

        # 6. Save experience / update
        sat['experience_buffer'].append((state, action_idx, reward, next_state, action_log_prob))
        sat['location_list'].append(action_obs["location"])  # already seen by this sat
        sat['rewards'].append(deepcopy(reward))

    def select_action(self, sat, state, num_actions, rand_action=False):
        # Default is random search
        random_idx = random.randint(0, num_actions - 1)
        return random_idx, None

    def get_satellite_state(self, sat):
        return [
            sat['sat_time'],
            sat['sat_angle'],
            sat['storage'],
            sat['sat_lat'],
            sat['sat_lon'],
        ]

    # ---------------------------------------------
    # Step methods
    # ---------------------------------------------

    def sim_step(self, satellites):
        took_action = self.sync_step(satellites)
        # took_action = self.async_step(satellites)
        return took_action

    def sync_step(self, satellites):
        for idx, sat in enumerate(satellites):
            debug = False
            if idx == 0 and self.step % 10 == 0:
                debug = True
            self.satellite_action(sat, debug)
        return True

    def async_step(self, satellites):
        action_taken = False
        if self.step == 0:
            for idx, sat in enumerate(satellites):
                self.satellite_action(sat, False)
                action_taken = True
        else:
            all_sat_times = [sat['sat_time'] for sat in satellites]
            max_sat_time = max(all_sat_times)
            max_sat_idx = all_sat_times.index(max_sat_time)
            for idx, sat in enumerate(satellites):
                if idx == max_sat_idx:
                    continue
                sat_time = sat['sat_time']
                debug = (self.step % 10 == 0) and (idx == 0)
                while sat_time < max_sat_time and sat['has_actions'] is True:
                    self.satellite_action(sat, debug)
                    sat_time = sat['sat_time']
                    action_taken = True
        return action_taken

    # ---------------------------------------------
    # Update models
    # ---------------------------------------------

    def update_satellite_models(self):
        pass

    # ---------------------------------------------
    # Plot progress
    # ---------------------------------------------

    def get_plot_items(self):
        return {
                'num_observations': self.num_observations,
                'mem_overflows': self.num_mem_overflows,
                'num_events_seen': self.num_events_seen,
                'num_unique_events_seen': self.num_unique_events_seen,
                'num_infeasibilities': self.num_infeasibilities,
                'num_dl_events': self.num_dl_events,
                'reward_history': self.reward_history,

                'total_observations': deepcopy(np.sum(self.num_observations)),
                'total_mem_overflows': deepcopy(np.sum(self.num_mem_overflows)),
                'total_events_seen': deepcopy(np.sum(self.num_events_seen)),
                'total_unique_events_seen': deepcopy(np.sum(self.num_unique_events_seen)),
                'total_infeasibilities': deepcopy(np.sum(self.num_infeasibilities)),
                'total_dl_events': deepcopy(np.sum(self.num_dl_events)),
                'total_points_seen': deepcopy(np.sum([gp['times_observed'] for gp in self.settings['grid_points']])),
                'actions_taken': deepcopy(([experience[1] for experience in self.satellites[0]['experience_buffer']])),
                'sat_0_rewards': deepcopy([reward for reward in self.satellites[0]['rewards']])
        }

    def plot_history(self):
        save_path = os.path.join(config.plots_dir, self.settings['name'], self.settings['planner'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if len(self.training_history) > 0:
            history = self.training_history[-1]
            epoch = len(self.training_history)
            self.plot_progress(history=history, epoch=epoch)

        # Episode wise plots
        if len(self.training_history) > 1:
            plot_path = os.path.join(save_path, self.settings['planner'] + '-results.png')
            epochs = [epoch+1 for epoch in range(len(self.training_history))]
            total_obs = [history['total_observations'] for history in self.training_history]
            total_mem_overflows = [history['total_mem_overflows'] for history in self.training_history]
            total_events = [history['total_events_seen'] for history in self.training_history]
            total_unique_events = [history['total_unique_events_seen'] for history in self.training_history]
            total_infeasibilities = [history['total_infeasibilities'] for history in self.training_history]
            total_dl_events = [history['total_dl_events'] for history in self.training_history]
            total_points_seen = [history['total_points_seen'] for history in self.training_history]
            total_actions_taken = [history['actions_taken'] for history in self.training_history]

            # Create a GridSpec object
            gs = gridspec.GridSpec(4, 2)

            fig = plt.figure(figsize=(8, 9))  # default [6.4, 4.8], W x H
            fig.suptitle(self.settings['name'] + ' | ' + self.settings['planner'], fontsize=16)
            # plt.subplot(4, 2, 1)
            plt.subplot(gs[0, 0])
            plt.plot(epochs, total_obs)
            plt.title("Total Observations")
            plt.xlabel('Epoch')
            plt.ylabel('Observations')

            # plt.subplot(4, 2, 2)
            plt.subplot(gs[0, 1])
            plt.plot(epochs, total_unique_events)  # Change to total unique events for paper
            plt.title("Total Unique Events")
            plt.xlabel('Epoch')
            plt.ylabel('Events')

            # # plt.subplot(4, 2, 3)
            # plt.subplot(gs[1, 0])
            # plt.plot(epochs, total_unique_events)
            # plt.title("Unique Events")
            # plt.xlabel('Epoch')
            # plt.ylabel('Events')
            plt.subplot(gs[1, 0])
            plt.plot(epochs, self.reward_history)
            plt.title("Reward Graph")
            plt.xlabel('Epoch')
            plt.ylabel('Reward')


            # plt.subplot(4, 2, 4)
            plt.subplot(gs[1, 1])
            plt.plot(epochs, total_infeasibilities)
            plt.title("Infeasible Points")
            plt.xlabel('Epoch')
            plt.ylabel('Points')

            # plt.subplot(gs[1, 1])
            # plt.plot(epochs, total_dl_events)
            # plt.title("Downlinked Events")
            # plt.xlabel('Epoch')
            # plt.ylabel('Events')


            # plt.subplot(4, 2, 5)
            plt.subplot(gs[2, 0])
            plt.plot(epochs, total_points_seen)
            plt.title("Total Points")
            plt.xlabel('Epoch')
            plt.ylabel('Points')


            # memory overflow actions taken
            # plt.subplot(4, 2, 6)
            plt.subplot(gs[2, 1])
            plt.plot(epochs, total_mem_overflows)
            plt.title("Total Memory Overflows")
            plt.xlabel('Epoch')
            plt.ylabel('MEM Overflows')
            plt.axhline(y=0, color='r', linestyle='--')

            # histogram of actions taken
            # plt.subplot(4, 2, 7)
            plt.subplot(gs[3, :])
            unique_actions = np.unique(np.concatenate(total_actions_taken))
            frequency_matrix = np.zeros((len(total_actions_taken), len(unique_actions)))
            for i, actions in enumerate(total_actions_taken):
                for action in actions:
                    j = np.where(unique_actions == action)[0][0]
                    frequency_matrix[i, j] += 1
            bar_width = 0.025
            index = np.arange(len(unique_actions))
            for i, frequency in enumerate(frequency_matrix):
                plt.bar(index + i * bar_width, frequency, width=bar_width, label=f"List {i + 1}")
            plt.title("Actions Taken")
            plt.xlabel("Action")
            plt.ylabel("Frequency")
            plt.xticks(index + bar_width, unique_actions)




            plt.tight_layout()
            plt.savefig(plot_path)
            plt.show()

    def plot_progress(self, history=None, epoch=None):
        if history is None:
            history = self.get_plot_items()

        save_path = os.path.join(config.plots_dir, self.settings['name'], self.settings['planner'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cumulative_obs = utils.accumulate(history['num_observations'])
        cumulative_events_seen = utils.accumulate(history['num_events_seen'])
        cumulative_unique_rewards = utils.accumulate(history['num_unique_events_seen'])
        cumulative_inf = utils.accumulate(history['num_infeasibilities'])
        cumulative_dl_events = utils.accumulate(history['num_dl_events'])

        cumulative_rewards = utils.accumulate(history['sat_0_rewards'])
        plotting_steps = [epoch+1 for epoch in range(len(cumulative_rewards))]

        plt.figure(figsize=(8, 6))
        plt.subplot(2, 2, 1)  # 3 rows, 1 column, 1st plot
        plt.plot(cumulative_obs, cumulative_events_seen)
        plt.title("Events Seen")
        plt.xlabel('Total Observations: ' + str(cumulative_obs[-1]))
        plt.ylabel('Events: ' + str(cumulative_events_seen[-1]))

        plt.subplot(2, 2, 2)  # 3 rows, 1 column, 2nd plot
        plt.plot(cumulative_obs, cumulative_unique_rewards)
        plt.title("Unique Events Seen")
        plt.xlabel('Total Observations: ' + str(cumulative_obs[-1]))
        plt.ylabel('Unique Events: ' + str(cumulative_unique_rewards[-1]))

        plt.subplot(2, 2, 3)  # 3 rows, 1 column, 3nd plot
        plt.plot(cumulative_obs, cumulative_inf)
        plt.title("Infeasible Points Missed")
        plt.xlabel('Total Observations: ' + str(cumulative_obs[-1]))
        plt.ylabel('Infeasible: ' + str(cumulative_inf[-1]))

        plt.subplot(2, 2, 4)  # 3 rows, 1 column, 3nd plot
        plt.plot(plotting_steps, cumulative_rewards)
        plt.title("Cumulative Rewards")
        plt.xlabel('Actions')
        plt.ylabel('Reward')

        if epoch is not None:
            plot_name = self.settings['planner'] + '-epoch-' + str(epoch) + '.png'
        else:
            plot_name = self.settings['planner'] + '-progress.png'
        plot_path = os.path.join(save_path, plot_name)

        # Show the plots
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # ---------------------------------------------
    # Get satellite plans
    # ---------------------------------------------

    def get_sat_plans(self):
        min_fov = np.min([self.settings["cross_track_ffov"], self.settings["along_track_ffov"]])

        # Gather satellite args
        satellites_with_args = [
            (sat["orbitpy_id"], sat['planned_obs'], min_fov, self.settings['event_locations'], sat['key']) for sat
            in self.satellites
        ]

        # Create a pool of worker processes
        num_processes = self.pool_size  # Get the number of CPU cores
        with Pool(num_processes) as pool:
            sat_plans = list(tqdm(pool.imap(AbstractPlanner.get_sat_plan, satellites_with_args), total=len(self.satellites),
                      desc='Recording Satellite Plans'))
        plan_dict = {}
        for sat_plan in sat_plans:
            plan_dict[sat_plan['key']] = sat_plan['plan']
        return plan_dict

    @staticmethod
    def get_sat_plan(args):
        orbit_id, all_obs, min_fov, grid_locations, key = args

        # Compute results
        results = []
        for obs in all_obs:
            in_fov = np.array([utils.within_fov_fast(loc, obs["location"], min_fov, 500) for loc in grid_locations])
            valid_locs = np.array(grid_locations)[in_fov]
            for loc in valid_locs:
                row = [obs["start"], obs["end"], loc[0], loc[1]]
                results.append(row)
        return {'plan': results, 'key': key}
