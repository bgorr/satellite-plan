import numpy as np
import os
import csv
import datetime
import multiprocessing
from functools import partial
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import time
import json
import pymap3d as pm
from datetime import datetime, timedelta
import math


def get_gridpoint(gp_index, settings):
    return settings['grid_points'][int(gp_index) + 1]


def close_enough(lat0, lon0, lat1, lon1):
    if np.sqrt((lat0 - lat1) ** 2 + (lon0 - lon1) ** 2) < 0.01:
        return True
    else:
        return False

def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))


def unique(lakes):
    lakes = np.asarray(lakes)
    return np.unique(lakes, axis=0)


def within_fov_fast(loc_array, loc_dict, angle, orbit_height_km):
    lat1, lon1 = np.deg2rad(loc_array)
    lat2, lon2 = np.deg2rad([loc_dict["lat"], loc_dict["lon"]])
    h = 0  # height above in m
    a = 6378e3  # m
    e = 0
    N_phi1 = a / np.sqrt(1 - e ** 2 * np.sin(lat1) ** 2)
    x1 = (N_phi1 + h) * np.cos(lat1) * np.cos(lon1)
    y1 = (N_phi1 + h) * np.cos(lat1) * np.sin(lon1)
    z1 = ((1 - e ** 2) * N_phi1 + h) * np.sin(lat1)
    N_phi2 = a / np.sqrt(1 - e ** 2 * np.sin(lat2) ** 2)
    x2 = (N_phi2 + h) * np.cos(lat2) * np.cos(lon2)
    y2 = (N_phi2 + h) * np.cos(lat2) * np.sin(lon2)
    z2 = ((1 - e ** 2) * N_phi2 + h) * np.sin(lat2)
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return np.arctan(dist / (orbit_height_km * 1e3)) < np.deg2rad(angle / 2)


def within_fov(loc_array, loc_dict, angle, orbit_height_km):

    lat1 = np.deg2rad(loc_array[0])
    lon1 = np.deg2rad(loc_array[1])
    lat2 = np.deg2rad(loc_dict["lat"])
    lon2 = np.deg2rad(loc_dict["lon"])
    h = 0  # height above in m
    a = 6378e3  # m
    e = 0
    N_phi1 = a / np.sqrt(1 - e ** 2 * np.sin(lat1) ** 2)
    x1 = (N_phi1 + h) * np.cos(lat1) * np.cos(lon1)
    y1 = (N_phi1 + h) * np.cos(lat1) * np.sin(lon1)
    z1 = ((1 - e ** 2) * N_phi1 + h) * np.sin(lat1)
    N_phi2 = a / np.sqrt(1 - e ** 2 * np.sin(lat2) ** 2)
    x2 = (N_phi2 + h) * np.cos(lat2) * np.cos(lon2)
    y2 = (N_phi2 + h) * np.cos(lat2) * np.sin(lon2)
    z2 = ((1 - e ** 2) * N_phi2 + h) * np.sin(lat2)
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    if np.arctan(dist / ((orbit_height_km) * 1e3)) < np.deg2rad(angle / 2):
        return True
    else:
        return False

def within_fov_full(lat1, lon1, lat2, lon2, angle, orbit_height_km):

    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    h = 0  # height above in m
    a = 6378e3  # m
    e = 0
    N_phi1 = a / np.sqrt(1 - e ** 2 * np.sin(lat1) ** 2)
    x1 = (N_phi1 + h) * np.cos(lat1) * np.cos(lon1)
    y1 = (N_phi1 + h) * np.cos(lat1) * np.sin(lon1)
    z1 = ((1 - e ** 2) * N_phi1 + h) * np.sin(lat1)
    N_phi2 = a / np.sqrt(1 - e ** 2 * np.sin(lat2) ** 2)
    x2 = (N_phi2 + h) * np.cos(lat2) * np.cos(lon2)
    y2 = (N_phi2 + h) * np.cos(lat2) * np.sin(lon2)
    z2 = ((1 - e ** 2) * N_phi2 + h) * np.sin(lat2)
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    if np.arctan(dist / ((orbit_height_km) * 1e3)) < np.deg2rad(angle / 2):
        return True
    else:
        return False


@tf.function
def tf_extract_from_batch(experiences, attribute):
    result = tf.py_function(extract_from_batch, [experiences, attribute], tf.float32)
    return result


def extract_from_batch(experiences, attribute):
    # experiences = experiences.numpy()
    # attribute = attribute.numpy().decode('utf-8')
    """
    Extracts an attribute (like 'state' or 'next_state') from a batch of experiences.
    Returns it as a list.
    """
    tensor = [getattr(exp, attribute) for exp in experiences]
    return tensor


def idx_from_batch(experiences, idx):
    return [exp[idx] for exp in experiences]


# ------------------------------------------------------------
# Speed-up: Faster computation of next 10 feasible observations
# - old: get_action_space
# - new: get_sat_action_space
# ------------------------------------------------------------


def get_action_space(curr_time, curr_angle, obs_list, last_obs, settings, top_n=5):
    feasible_actions = []
    for obs in obs_list:
        if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
            continue
        if obs["start"] > curr_time:
            feasible, transition_end_time = check_maneuver_feasibility(
                curr_angle,
                np.min(obs["angles"]),
                curr_time,
                obs["end"],
                settings
            )
            if transition_end_time < obs["start"]:
                obs["soonest"] = obs["start"]
            else:
                obs["soonest"] = transition_end_time
            if feasible:
                feasible_actions.append(obs)
        if len(feasible_actions) >= top_n:
            break
    return feasible_actions


def get_action_space_info(curr_time, curr_angle, obs_list, last_obs, settings, top_n=5):
    feasible_actions = []
    num_infeasible = 0
    for obs in obs_list:
        if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
            continue
        if obs["start"] > curr_time:
            feasible, transition_end_time = check_maneuver_feasibility(
                curr_angle,
                np.min(obs["angles"]),
                curr_time,
                obs["end"],
                settings
            )
            if transition_end_time < obs["start"]:
                obs["soonest"] = obs["start"]
            else:
                obs["soonest"] = transition_end_time

            if feasible:
                feasible_actions.append(obs)
            else:
                num_infeasible += 1
        if len(feasible_actions) >= top_n:
            break
    return feasible_actions, num_infeasible


def get_obs_idx(curr_time, all_obs):
    for idx, obs in enumerate(all_obs):
        if obs["start"] > curr_time:
            return idx
    return len(all_obs) - 1


def get_obs_actions(sat, settings, top_n=5):
    curr_time = sat['sat_time']
    curr_angle = sat['sat_angle']
    obs_list = sat['all_obs']
    last_obs = sat['last_obs']

    # 1. Get next n possible observations
    feasible_actions = []
    for obs in obs_list:
        if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
            continue
        if obs["start"] > curr_time:
            feasible, transition_end_time = check_maneuver_feasibility(
                curr_angle,
                np.min(obs["angles"]),
                curr_time,
                obs["end"],
                settings
            )
            if transition_end_time < obs["start"]:
                obs["soonest"] = obs["start"]
            else:
                obs["soonest"] = transition_end_time

            if feasible:
                # Set reward / next storage state
                if sat['storage'] >= settings['sat-storage-cap']:
                    obs['reward'] = settings['mem-overflow-penalty']
                    obs['storage'] = deepcopy(sat['storage'])
                    obs['stored_image'] = False
                else:
                    obs['reward'] = 0
                    obs['storage'] = sat['storage'] + 1
                    obs['stored_image'] = True

                feasible_actions.append(obs)
        if len(feasible_actions) >= top_n:
            break
    return feasible_actions


def get_dl_actions(sat, settings):
    # last_obs = sat['last_obs']
    # if last_obs and last_obs['type'] == 'down-link':
    #     return []

    # - There should always exist a down-link action (except when no GS left in mission)
    # 1. Determine if GS in sight
    # 2. If GS in sight:
    #   - Determine max down-link window
    downlink_action = deepcopy({
        "angle": sat['sat_angle'],
        "type": "down-link",
        "reward": 0,  # calculate reward
        "location": {
            "lat": 0.0,
            "lon": 0.0
        }
    })

    dl_steps_per_image = settings['dl-steps-per-image']
    curr_time = sat['sat_time']
    gs_obs = sat['all_gs']
    gs_obs.sort(key=lambda x: x[0])
    gs_in_view = [(gs[0] <= curr_time < gs[1]) for gs in gs_obs]

    # IF: GS in view
    if True in gs_in_view:
        next_gs = gs_obs[gs_in_view.index(True)]
        available_downlink_time = next_gs[1] - curr_time
        dl_start = curr_time
        # print('--> GS IN VIEW', available_downlink_time)

    else:  # ELSE: GS not in view
        future_gs = [gs for gs in gs_obs if gs[0] > curr_time]
        if len(future_gs) == 0:
            return []
        next_gs = future_gs[0]
        available_downlink_time = next_gs[1] - next_gs[0]
        dl_start = next_gs[0]
        # print('--> GS NOT IN VIEW', available_downlink_time)

    downlinkable_images = math.floor(available_downlink_time / dl_steps_per_image)


    downlink_action['location'] = {
        'lat': next_gs[2],
        'lon': next_gs[3]
    }
    # print('----> CURR TIME | NEXT GS:', curr_time, next_gs[0], next_gs[1])

    # IF: can downlink all images
    if downlinkable_images >= sat['storage']:
        downlink_size = deepcopy(sat['storage'])
        downlink_action['storage'] = 0
    else:  # ELSE: can downlink some images
        downlink_size = downlinkable_images
        downlink_action['storage'] = deepcopy(sat['storage'] - downlink_size)

    dl_reward = 0
    downlink_action['storage_img_type'] = deepcopy(sat['storage_img_type'])
    for img_num in range(downlink_size):
        img = downlink_action['storage_img_type'].pop(0)
        if img == 'N':
            dl_reward += settings['dl-obs-reward']
        elif img == 'E':
            dl_reward += settings['dl-event-reward']
        else:
            raise ValueError('--> UNKNOWN IMAGE TYPE')

    # last_obs = sat['last_obs']
    # if last_obs and last_obs['type'] == 'down-link' and dl_reward == 0:
    #     dl_reward = 1.0 * settings['dl-nothing-reward']

    downlink_action['reward'] = dl_reward




    downlink_action['soonest'] = dl_start + downlink_size * dl_steps_per_image
    if downlinkable_images == 0 or downlink_size == 0:  # Fast forward to end of GS if window too small for any downlink
        downlink_action['soonest'] = next_gs[1]

    return [downlink_action]





def check_maneuver_feasibility(curr_angle, obs_angle, curr_time, obs_end_time, settings):
    """
    Checks to see if the specified angle change violates the maximum slew rate constraint.
    """
    if (obs_end_time == curr_time):
        return False, False
    slew_rate = abs(obs_angle - curr_angle) / abs(obs_end_time - curr_time) / settings["step_size"]
    max_slew_rate = settings["agility"]  # deg / s
    transition_end_time = abs(obs_angle - curr_angle) / (max_slew_rate * settings["step_size"]) + curr_time
    return slew_rate < max_slew_rate, transition_end_time


def get_sat_action_space(curr_time, curr_angle, obs_list, last_obs, settings):
    feasible_actions = []
    if last_obs:
        last_obs_lat = {last_obs["location"]["lat"]}  # Using a set for faster lookup
    agility = settings["agility"]
    step_size = settings["step_size"]
    for obs in obs_list:
        if last_obs and obs["location"]["lat"] in last_obs_lat:
            continue
        if obs["start"] > curr_time:
            feasible, transition_end_time = check_sat_maneuver_feasibility(
                curr_angle, np.min(obs["angles"]), curr_time, obs["end"], agility, step_size
            )
            obs["soonest"] = max(obs["start"], transition_end_time)
            if feasible:
                feasible_actions.append(obs)
            if len(feasible_actions) >= 10:  # Break out early if 10 feasible actions are found
                break

    return feasible_actions


def check_sat_maneuver_feasibility(curr_angle, obs_angle, curr_time, obs_end_time, agility, step_size):
    if obs_end_time == curr_time:
        return False, False
    slew_rate = abs(obs_angle - curr_angle) / ((obs_end_time - curr_time) * step_size)
    transition_end_time = abs(obs_angle - curr_angle) / (agility * step_size) + curr_time
    return slew_rate < agility, transition_end_time

# ------------------------------------------------------------
# Speed-up: Faster computation of satellite observations
# - old: init_observations
# - new: init_sat_observations
# ------------------------------------------------------------


def init_observations(visibilities):
    obs_list = []
    i = 0
    while i < len(visibilities):
        continuous_visibilities = []
        visibility = visibilities[i]
        continuous_visibilities.append(visibility)
        start = visibility[0]
        end = visibility[0]
        while (i < len(visibilities) - 1 and visibilities[i + 1][0] == start):
            i += 1
        vis_done = False
        if i == len(visibilities) - 1:
            break
        while not vis_done:
            vis_done = True
            num_steps = len(continuous_visibilities)
            while visibilities[i + 1][0] == start + num_steps:
                if visibilities[i + 1][1] == visibility[1]:
                    continuous_visibilities.append(visibilities[i + 1])
                    end = visibilities[i + 1][0]
                    vis_done = False
                if i == len(visibilities) - 2:
                    break
                else:
                    i += 1
            num_steps = len(continuous_visibilities)
            if i == len(visibilities) - 1:
                break
        time_window = {
            "location": {
                "lat": visibility[3],
                "lon": visibility[4]
            },
            "times": [x[0] for x in continuous_visibilities],
            "angles": [x[6] for x in continuous_visibilities],
            "start": start,
            "end": end,
            "angle": visibility[6],
            "gp_index": visibility[1],
            "reward": 0
        }
        if (time_window["location"]) is None:
            print(time_window)
        obs_list.append(time_window)
        for cont_vis in continuous_visibilities:
            visibilities.remove(cont_vis)
        i = 0

    obs_list.sort(key=lambda x: x['start'])
    return obs_list

def init_sat_observations_proc(visibilities):
    ''' Visibility Structure
        visible_point = [
            1.0,     # time index
            3322.0,  # GP index
            0.0,     # pnt-opt index
            3.496,   # lat [deg]
            -99.38,  # lon [deg]
            605.7,   # observation range [km]
            -33.04,  # look angle [deg]
            36.01,   # incidence angle [deg]
            82.11    # solar zenith [deg]
        ]
    '''

    visibilities.sort(key=lambda x: (x[1], x[0]))

    windows = []
    viz_windows = []
    continuous_visibilities = []
    prev_time_index = None
    prev_gp_index = None

    for visibility in visibilities:
        current_time_index = visibility[0]
        current_gp_index = visibility[1]

        if prev_gp_index is not None and (
                prev_gp_index != current_gp_index or current_time_index - prev_time_index > 1):

            time_window = compute_time_window(continuous_visibilities)
            viz_window = compute_viz_window(continuous_visibilities)
            windows.append(time_window)
            viz_windows.append(viz_window)

            # Reset continuous visibilities list
            continuous_visibilities = []

        continuous_visibilities.append(visibility)
        prev_time_index = current_time_index
        prev_gp_index = current_gp_index

    windows.sort(key=lambda x: x['start'])
    # windows.sort(key=lambda x: (x['start'], abs(x['angle'])))
    viz_windows.sort(key=lambda x: x[0])


    return {
        'windows': windows,
        'viz_windows': viz_windows
    }

def init_sat_observations(visibilities):
    ''' Visibility Structure
        visible_point = [
            1.0,     # time index
            3322.0,  # GP index
            0.0,     # pnt-opt index
            3.496,   # lat [deg]
            -99.38,  # lon [deg]
            605.7,   # observation range [km]
            -33.04,  # look angle [deg]
            36.01,   # incidence angle [deg]
            82.11    # solar zenith [deg]
        ]
    '''

    visibilities.sort(key=lambda x: (x[1], x[0]))

    windows = []
    continuous_visibilities = []
    prev_time_index = None
    prev_gp_index = None

    for visibility in visibilities:
        current_time_index = visibility[0]
        current_gp_index = visibility[1]

        if prev_gp_index is not None and (
                prev_gp_index != current_gp_index or current_time_index - prev_time_index > 1):

            time_window = compute_time_window(continuous_visibilities)
            windows.append(time_window)

            # Reset continuous visibilities list
            continuous_visibilities = []

        continuous_visibilities.append(visibility)
        prev_time_index = current_time_index
        prev_gp_index = current_gp_index

    windows.sort(key=lambda x: x['start'])
    # windows.sort(key=lambda x: (x['start'], abs(x['angle'])))
    return windows


def compute_time_window(continuous_visibilities):
    first_visibility = continuous_visibilities[0]
    last_visibility = continuous_visibilities[-1]
    time_window = {
        "location": {
            "lat": first_visibility[3],
            "lon": first_visibility[4]
        },
        "times": [x[0] for x in continuous_visibilities],
        "angles": [x[6] for x in continuous_visibilities],
        "start": first_visibility[0],
        "end": last_visibility[0],
        "angle": first_visibility[6],
        "gp_index": first_visibility[1],
        "reward": 0,
        "type": "observation"
    }
    return time_window


def init_sat_observations_post_proc(visibilities):
    visibilities.sort(key=lambda x: (x[1], x[0]))

    windows = []
    continuous_visibilities = []
    prev_time_index = None
    prev_gp_index = None

    for visibility in visibilities:
        current_time_index = visibility[0]
        current_gp_index = visibility[1]

        if prev_gp_index is not None and (
                prev_gp_index != current_gp_index or current_time_index - prev_time_index > 1):
            # Add time window for the previous group of continuous visibilities
            time_window = compute_viz_window(continuous_visibilities)
            windows.append(time_window)

            # Reset continuous visibilities list
            continuous_visibilities = []

        continuous_visibilities.append(visibility)
        prev_time_index = current_time_index
        prev_gp_index = current_gp_index

    windows.sort(key=lambda x: x[0])
    return windows


def compute_viz_window(continuous_visibilities):
    first_visibility = continuous_visibilities[0]
    last_visibility = continuous_visibilities[-1]
    vis_window = [first_visibility[0], last_visibility[0], first_visibility[3], first_visibility[4], first_visibility[-1]]
    return vis_window

# --------------------------------------
# Event Bonus
# --------------------------------------


def find_event_bonus(best_obs, satellite, events):
    for event in events:
        if close_enough(best_obs["location"]["lat"], best_obs["location"]["lon"], event["location"]["lat"], event["location"]["lon"]):
            if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                obs_time = best_obs['start']
                base_reward = event['reward']
                reward = base_reward
                return reward
    return 0.0

def find_event_bool(best_obs, events):
    for event in events:
        e_start = float(event[2])
        e_end = float(event[2]) + float(event[3])
        e_lat = float(event[0])
        e_lon = float(event[1])
        if close_enough(best_obs["location"]["lat"], best_obs["location"]["lon"], e_lat, e_lon):
            if (e_start <= best_obs["start"] <= e_end) or (e_start <= best_obs["end"] <= e_end):
                return 1
    return 0




def abstract_event_bonus(best_obs, satellite, events):
    event_seen = 0
    novel_event = 0
    for event in events:
        if close_enough(best_obs["location"]["lat"], best_obs["location"]["lon"], event["location"]["lat"], event["location"]["lon"]):
            if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                obs_time = best_obs['start']
                base_reward = event['reward']
                event_seen = 1
                if event['times_observed'] == 0:
                    novel_event = 1
                event['times_observed'] += 1


                # Reward Func Terms
                # 1. Temporal penalty: add penalty for time until first obs (max time val ~ 8.6k for one day)
                # 2. Discovery bonus: add bonus for newly discovered events
                # 3. Rarity bonus: add bonus for rare events
                # 4. Collaboration bonus: add bonus for multiple satellites observing the same event

                temporal_penalty = (obs_time / 10000.0) * -1.0

                # sat_discovery_time = event['satellite_discoveries'][satellite['sat_id']]
                # if sat_discovery_time == -1:
                #     event['satellite_discoveries'][satellite['sat_id']] = obs_time
                #     event['times_observed'] += 1.0
                #     discovery_bonus = 1.0
                # else:
                #     discovery_bonus = 0.0
                # rarity_bonus = 1.0 / event['rarity']
                # collaboration_bonus = 1 - (1.0 / event['times_observed'])

                reward = base_reward
                return reward, event_seen, novel_event
    return 0.0, event_seen, novel_event


def full_event_bonus(satellite, next_obs, settings):
    min_fov = np.min([settings["cross_track_ffov"], settings["along_track_ffov"]])
    next_obs_lat = next_obs["location"]["lat"]
    next_obs_lon = next_obs["location"]["lon"]

    reward = 0.0
    novel_counter = 0
    event_counter = 0
    for event in settings['events']:
        event_lat = event["location"]["lat"]
        event_lon = event["location"]["lon"]
        in_view = within_fov_full(event_lat, event_lon, next_obs_lat, next_obs_lon, min_fov, 500)
        in_time_window = (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"])
        if in_view and in_time_window:
            reward += event['reward']
            event_counter += 1
            if event['times_observed'] == 0:
                novel_counter += 1
            event['times_observed'] += 1

    for grid_point in settings['grid_points']:
        gp_lat = grid_point["location"]["lat"]
        gp_lon = grid_point["location"]["lon"]
        in_view = within_fov_full(gp_lat, gp_lon, next_obs_lat, next_obs_lon, min_fov, 500)
        if in_view:
            grid_point['times_observed'] += 1


    return reward, event_counter, novel_counter


def obs_event_bonus(satellite, next_obs, settings):
    all_obs = satellite['all_obs']
    curr_time = satellite['sat_time']
    next_obs_time = next_obs['soonest']
    delta_t = abs(next_obs_time - curr_time)
    event_counter = 0
    novel_counter = 0
    reward = 0.0
    event_seen = False
    for idx, obs in enumerate(all_obs):
        if obs == next_obs:
            gp_index = int(obs['gp_index'])
            g_point = settings['grid_points'][gp_index + 1]
            g_point_lat = g_point['location']['lat']
            g_point_lon = g_point['location']['lon']
            g_point['times_observed'] += 1
            for event in g_point['events']:
                if event['start'] <= next_obs_time <= event['end']:
                    event_counter += 1
                    event_seen = True
                    if event['times_observed'] == 0:
                        novel_counter += 1
                    event['times_observed'] += 1
                    reward += event['reward']

    if next_obs['stored_image'] is True:
        if event_seen is True:
            satellite['storage_img_type'].append('E')
        else:
            satellite['storage_img_type'].append('N')

    return reward, event_counter, novel_counter


def transition_event_bonus(satellite, next_obs, settings):
    if not satellite['last_obs']:
        reward, event_counter, novel_counter = full_event_bonus(satellite, next_obs, settings)
        return reward, event_counter, novel_counter, 0.0


    min_fov = np.min([settings["cross_track_ffov"], settings["along_track_ffov"]])

    curr_time = satellite['sat_time']
    curr_angle = satellite['sat_angle']
    curr_obs_lat = satellite['last_obs']['location']['lat']
    curr_obs_lon = satellite['last_obs']['location']['lon']

    next_time = next_obs["soonest"]
    next_angle = next_obs["angle"]
    next_obs_lat = next_obs["location"]["lat"]
    next_obs_lon = next_obs["location"]["lon"]

    slew_time_steps = []
    interpolated_lats = []  # start with the current latitude
    interpolated_lons = []  # start with the current longitude

    # 1. Interpolate lat/lons to check for intermediate event sightings
    time_cnt = deepcopy(curr_time)
    while time_cnt < next_time:
        time_cnt += 0.2
        slew_time_steps.append(deepcopy(time_cnt))

        # Interpolate latitude
        interpolated_lat = curr_obs_lat + ((next_obs_lat - curr_obs_lat) / (next_time - curr_time)) * (
                    time_cnt - curr_time)
        interpolated_lats.append(interpolated_lat)

        # Interpolate longitude
        interpolated_lon = curr_obs_lon + ((next_obs_lon - curr_obs_lon) / (next_time - curr_time)) * (
                    time_cnt - curr_time)
        interpolated_lons.append(interpolated_lon)

    slew_time_steps.append(next_time)
    interpolated_lats.append(next_obs_lat)
    interpolated_lons.append(next_obs_lon)


    # 2. Gather viewable set of observations
    curr_range = [curr_time, next_time]
    all_obs = satellite['all_obs']
    valid_obs = []
    for idx, obs in enumerate(all_obs):
        if obs == next_obs:
            valid_obs.append(obs)
            continue
        if obs == satellite['last_obs']:
            continue
        if obs in satellite['all_last_obs']:
            continue
        obs_lat, obs_lon = obs['location']['lat'], obs['location']['lon']
        if obs_lat == curr_obs_lat and obs_lon == curr_obs_lon:
            continue
        a, b = obs['start'], obs['end']
        c, d = curr_range
        if a <= d and b >= c:
            valid_obs.append(obs)
    valid_obs_seen = [False] * len(valid_obs)

    # 3. Check for event sightings
    delta_t = abs(curr_time - next_time)

    all_next_obs = []
    reward = 0.0
    obs_counter = 0
    event_counter = 0
    novel_counter = 0
    for i_lat, i_lon, i_time in zip(interpolated_lats, interpolated_lons, slew_time_steps):
        for idx, v_obs in enumerate(valid_obs):
            if valid_obs_seen[idx] is True:
                continue
            obs_lat = v_obs['location']['lat']
            obs_lon = v_obs['location']['lon']
            in_view = within_fov_full(obs_lat, obs_lon, i_lat, i_lon, min_fov, 500)
            if in_view is True:
                all_next_obs.append(v_obs)
                valid_obs_seen[idx] = True
                gp_index = int(v_obs['gp_index'])
                g_point = settings['grid_points'][gp_index + 1]
                g_point_lat = g_point['location']['lat']
                g_point_lon = g_point['location']['lon']
                g_point['times_observed'] += 1
                obs_counter += 1
                for event in g_point['events']:
                    if event['start'] <= i_time <= event['end']:
                        event_counter += 1
                        if event['times_observed'] == 0:
                            novel_counter += 1
                        event['times_observed'] += 1
                        reward += event['reward']

    satellite['all_last_obs'] = all_next_obs
    # print('OBSERVATIONS / EVENTS / NOVEL EVENTS: ', obs_counter, event_counter, novel_counter, len(slew_time_steps))
    return (reward / delta_t), event_counter, novel_counter, (obs_counter / delta_t)


# ----------------------------------------------------
# Recording results
# ----------------------------------------------------


def record_json_results(overall_results, settings, elapsed_time, epoch):
    # Create a dictionary to store the data
    data = {
        "name": settings["name"],
        "for": settings["ffor"],
        "fov": settings["ffov"],
        "constellation_size": settings["constellation_size"],
        "agility": settings["agility"],
        "event_duration": settings["event_duration"],
        "event_frequency": settings["event_frequency"],
        "event_density": settings["event_density"],
        "event_clustering": settings["event_clustering"],
        "planner": settings["planner"],
        "reobserve": settings["planner_options"]["reobserve"],
        "reward": settings["reward"],

        # Overall
        "events": overall_results["num_events"],

        # Total observations
        "satellite observations": overall_results["num_obs_init"],
        "possible satellite observations": overall_results["num_vis"],

        # Event observations
        "event observations": overall_results["init_results"]["event_obs_count"],
        "possible event observations": overall_results["vis_results"]["event_obs_count"],

        # Events seen at least once
        "events seen once": overall_results["init_results"]["events_seen_once"],
        "possible events seen once": overall_results["vis_results"]["events_seen_once"],


        "time": elapsed_time
    }

    # delete file if exists
    if os.path.exists(f"./reward_comparison_{str(epoch)}.json"):
        os.remove(f"./reward_comparison_{str(epoch)}.json")

    # Save the dictionary as a JSON file
    with open(f"./reward_comparison_{str(epoch)}.json", 'w') as json_file:
        json.dump(data, json_file, indent=4)  # The `indent` parameter makes it nicely formatted


def record_results(overall_results, settings, elapsed_time, epoch):
    with open(f"./reward_comparison_{str(epoch)}.csv", 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        first_row = ["name", "for", "fov", "constellation_size", "agility",
                     "event_duration", "event_frequency", "event_density", "event_clustering",
                     "planner", "reobserve", "reward",
                     "events", "init_obs_count", "replan_obs_count", "vis_count",
                     "init_event_obs_count", "init_events_seen",
                     "replan_event_obs_count", "replan_events_seen",
                     "vis_event_obs_count", "vis_events_seen", "time"]
        csvwriter.writerow(first_row)
        row = [
            settings["name"],
            settings["ffor"],
            settings["ffov"],
            settings["constellation_size"],
            settings["agility"],
            settings["event_duration"],
            settings["event_frequency"],
            settings["event_density"],
            settings["event_clustering"],
            settings["planner"],
            settings["planner_options"]["reobserve"],
            settings["reward"],

            # Overall
            overall_results["num_events"],
            overall_results["num_obs_init"],
            overall_results["num_obs_replan"],
            overall_results["num_vis"],

            # Executed Observations
            overall_results["init_results"]["event_obs_count"],
            overall_results["init_results"]["events_seen_once"],

            # Potential Observations
            overall_results["vis_results"]["event_obs_count"],
            overall_results["vis_results"]["events_seen_once"],
            elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()


# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------


def seconds_to_datetime(seconds_since):
    reference_date = datetime(2020, 1, 1, 0, 0, 0)
    return reference_date + timedelta(seconds=seconds_since)


def eci_to_latlon(x, y, z, seconds_since):
    target_date = seconds_to_datetime(seconds_since)

    # Convert ECI to ECEF
    x_ecef, y_ecef, z_ecef = pm.eci2ecef(x, y, z, target_date)

    # Convert ECEF to lat/lon/alt
    lat, lon, _ = pm.ecef2geodetic(x_ecef, y_ecef, z_ecef)

    return lat, lon








def accumulate(values):
    accumulation = []
    curr_sum = 0
    for val in values:
        curr_sum += val
        accumulation.append(deepcopy(curr_sum))
    return accumulation