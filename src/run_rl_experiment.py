import datetime
import os
import numpy as np
import csv
import time

from create_mission import create_mission
from execute_mission import execute_mission
from process_mission import process_mission
from planners.BaseRL import BaseRL
from plot_mission_cartopy import plot_mission
from planners.utils import record_results, record_json_results

from utils.convert_geo import convert_geo_cords
from results.ExperimentResult import ExperimentResult


def run():
    default_settings = {
        "name": "rl_default",
        "ffor": 60,
        "ffov": 5,
        "constellation_size": 6,
        "agility": 1,
        "event_duration": 6 * 3600,  # 6 hours event duration (seconds)
        "event_frequency": 0.01 / 3600,  # probability event gets created at each location per time-step
        "event_density": 10,  # points consideresd per 10 deg lat/lon grid cell
        "event_clustering": 4,  # specifies clustering of points in lat/lon grid cells (var of gaussian dist)
        "planner": "rl",
        "planner_options": {
            "reobserve": "encouraged",
            "reobserve_reward": 2
        },
        "reward": 0
    }

    # 1. Run experiment
    start = time.time()
    overall_results = run_experiment(default_settings)
    elapsed_time = time.time() - start

    # 2. Record results
    for epoch, result in enumerate(overall_results):
        record_json_results(result, default_settings, elapsed_time, epoch)


def run_experiment(experiment_settings):

    # ------------------------------------------------------------
    # Extract parameters
    # ------------------------------------------------------------
    seconds_per_day = 86400

    simulation_step_size = 10  # seconds
    simulation_duration = 1  # days
    mission_name = experiment_settings["name"]
    cross_track_ffor = experiment_settings["ffor"]
    along_track_ffor = experiment_settings["ffor"]
    cross_track_ffov = experiment_settings["ffov"]
    along_track_ffov = experiment_settings["ffov"]  # TODO carefully consider this assumption
    agility = experiment_settings["agility"]
    num_planes = experiment_settings["constellation_size"]
    num_sats_per_plane = experiment_settings["constellation_size"]
    var = experiment_settings["event_clustering"]
    num_points_per_cell = experiment_settings["event_density"]
    event_frequency = experiment_settings["event_frequency"]
    event_duration = experiment_settings["event_duration"]

    # Generate time steps
    steps = np.arange(0, simulation_duration * seconds_per_day, simulation_step_size)

    # ------------------------------------------------------------
    # Create coverage grid (10deg x 10deg lat/lon)
    # Determine all possible event locations (lat/lon pairs)
    # ------------------------------------------------------------

    if not os.path.exists("./coverage_grids/" + mission_name + "/event_locations.csv"):
        event_locations = []
        center_lats = np.arange(-85, 95, 10)
        center_lons = np.arange(-175, 185, 10)
        for clat in center_lats:
            for clon in center_lons:
                mean = [clat, clon]
                cov = [[var, 0], [0, var]]
                xs, ys = np.random.multivariate_normal(mean, cov, num_points_per_cell).T
                for i in range(len(xs)):
                    location = [xs[i], ys[i]]
                    event_locations.append(location)
        if not os.path.exists("./coverage_grids/" + mission_name + "/"):
            os.mkdir("./coverage_grids/" + mission_name + "/")
        with open("./coverage_grids/" + mission_name + "/event_locations.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]', 'lon [deg]'])
            for location in event_locations:
                csvwriter.writerow(location)

    # ------------------------------------------------------------
    # Stochastically determine true events and record
    # - events: (lat / lon pairs, time-step, duration, ---)
    # ------------------------------------------------------------

    if not os.path.exists("./events/" + mission_name + "/events.csv"):
        events = []
        for step in steps:
            for location in event_locations:  # event lat / lon pair
                if np.random.random() < event_frequency * simulation_step_size:
                    event = [location[0], location[1], step, event_duration, 1]
                    events.append(event)
        if not os.path.exists("./events/" + mission_name + "/"):
            os.mkdir("./events/" + mission_name + "/")
        with open("./events/" + mission_name + "/events.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]', 'lon [deg]', 'start time [s]', 'duration [s]', 'severity'])
            for event in events:
                csvwriter.writerow(event)

    # ------------------------------------------------------------
    # Update settings / paths
    # ------------------------------------------------------------

    settings = {
        "directory": "./missions/" + mission_name + "/",
        "step_size": simulation_step_size,
        "duration": simulation_duration,
        "initial_datetime": datetime.datetime(2020, 1, 1, 0, 0, 0),
        "grid_type": "event",  # can be "event" or "static"
        "point_grid": "./coverage_grids/" + mission_name + "/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/" + mission_name + "/events.csv"],
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False,
        "planner": experiment_settings["planner"],
        "planner_options": experiment_settings["planner_options"],
        "experiment_settings": experiment_settings
    }
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])

    # ------------------------------------------------------------
    # Create and execute mission
    # ------------------------------------------------------------

    if not os.path.exists(settings["directory"] + 'orbit_data/'):
        os.mkdir(settings["directory"] + 'orbit_data/')
        create_mission(settings)
        execute_mission(settings)

    # ------------------------------------------------------------
    # Calc geo coords
    # ------------------------------------------------------------
    convert_geo_cords(settings)


    # ------------------------------------------------------------
    # Planning Step
    # ------------------------------------------------------------
    num_epochs = 1
    # planner = RLMissionPlanner(settings)
    planner = BaseRL(settings)
    planner.train_planners(num_epochs=num_epochs)

    # plan_mission(settings)  # must come before process as process expects a plan.csv in the orbit_data directory
    # plan_mission_replan_interval(settings)

    # overall_results = compute_experiment_statistics(settings)
    epoch_results = []
    result_client = ExperimentResult(settings)
    for epoch in range(num_epochs):
        epoch_result = result_client.run(epoch)
        epoch_results.append(epoch_result)

    return epoch_results


if __name__ == "__main__":
    run()



















