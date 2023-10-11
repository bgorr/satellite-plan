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


def close_enough(lat0, lon0, lat1, lon1):
    if np.sqrt((lat0 - lat1) ** 2 + (lon0 - lon1) ** 2) < 0.01:
        return True
    else:
        return False


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

def extract_from_batch(experiences, attribute):
    """
    Extracts an attribute (like 'state' or 'next_state') from a batch of experiences.
    Returns it as a list.
    """
    return [getattr(exp, attribute) for exp in experiences]

# ------------------------------------------------------------
# Speed-up: Faster computation of next 10 feasible observations
# - old: get_action_space
# - new: get_sat_action_space
# ------------------------------------------------------------


def get_action_space(curr_time, curr_angle, obs_list, last_obs, settings):
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
        if len(feasible_actions) >= 10:
            break
    return feasible_actions


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
        "reward": 0
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




