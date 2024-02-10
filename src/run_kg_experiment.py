import datetime
import os
import numpy as np
import csv
import time
import shutil, errno
import random

from create_mission import create_mission
from execute_mission import execute_mission
from plan_mission_kg import plan_mission, plan_mission_replan_interval
from utils.compute_experiment_statistics_kg import compute_experiment_statistics

def run_experiment(settings):
    simulation_step_size = settings["time"]["step_size"] # seconds
    simulation_duration = settings["time"]["duration"] # days
    mission_name = settings["name"]
    var = settings["events"]["event_clustering"]
    num_points_per_cell = settings["events"]["event_density"]
    event_frequency = settings["events"]["event_frequency"]
    event_duration = settings["events"]["event_duration"]
    steps = np.arange(0,simulation_duration*86400,simulation_step_size)
    if not os.path.exists("./coverage_grids/"+mission_name+"/event_locations.csv"):
        event_locations = []
        center_lats = np.arange(-85,95,10)
        center_lons = np.arange(-175,185,10)
        for clat in center_lats:
            for clon in center_lons:
                mean = [clat, clon]
                cov = [[var, 0], [0, var]]
                xs, ys = np.random.multivariate_normal(mean, cov, num_points_per_cell).T
                for i in range(len(xs)):
                    location = [xs[i],ys[i]]
                    event_locations.append(location)
        if not os.path.exists("./coverage_grids/"+mission_name+"/"):
                os.mkdir("./coverage_grids/"+mission_name+"/")
        with open("./coverage_grids/"+mission_name+"/event_locations.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]'])
            for location in event_locations:
                csvwriter.writerow(location)
    
    if not os.path.exists("./events/"+mission_name+"/events.csv"):
        events = []
        variables = ['Land surface temperature','Ocean Salinity','Ozone','Sea surface temperature','Sea-ice cover','Cloud cover','Cloud base height','Geoid','Land surface imagery']
        for location in event_locations:
            if np.random.random() < event_frequency*simulation_step_size:
                event = [location[0],location[1],0,86400,1,random.choice(variables)]
                events.append(event)
        if not os.path.exists("./events/"+mission_name+"/"):
            os.mkdir("./events/"+mission_name+"/")
        with open("./events/"+mission_name+"/events.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity','parameter'])
            for event in events:
                csvwriter.writerow(event)

    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
        create_mission(settings)
        execute_mission(settings)
    plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    #plan_mission_replan_interval(settings)
    overall_results = compute_experiment_statistics(settings)
    return overall_results


if __name__ == "__main__":
    with open('./kg_test.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        first_row = ["name","for","fov","constellation_size","agility",
                    "event_duration","event_frequency","event_density","event_clustering","num_meas_types",
                    "planner","reobserve_reward", "reward","time"]
        csvwriter.writerow(first_row)
        csvfile.close()
    name = "kg_test"
    settings = {
        "name": name,
        "instrument": {
            "ffor": 30,
            "ffov": 0
        },
        "agility": {
            "slew_constraint": "rate",
            "max_slew_rate": 0.1,
            "inertia": 2.66,
            "max_torque": 4e-3
        },
        "orbit": {
            "altitude": 705, # km
            "inclination": 98.4, # deg
            "eccentricity": 0.0001,
            "argper": 0, # deg
        },
        "constellation": {
            "num_sats_per_plane": 3,
            "num_planes": 3,
            "phasing_parameter": 1
        },
        "events": {
            "event_duration": 3600*6,
            "event_frequency": 0.01/3600,
            "event_density": 2,
            "event_clustering": 4
        },
        "time": {
            "step_size": 10, # seconds
            "duration": 1, # days
            "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
        },
        "rewards": {
            "reward": 10,
            "reward_increment": 0.1,
            "reobserve_reward": 2
        },
        "planner": "dp",
        "num_meas_types": 3,
        "sharing_horizon": 1000,
        "planning_horizon": 1000,
        "directory": "./missions/"+name+"/",
        "grid_type": "custom", # can be "uniform" or "custom"
        "point_grid": "./coverage_grids/"+name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+name+"/events.csv"],
        "process_obs_only": False,
        "conops": "onboard_processing"
    }
    start = time.time()
    print(settings["name"])
    overall_results = run_experiment(settings)
    end = time.time()
    elapsed_time = end-start
    with open('./kg_test.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        row = [settings["name"],settings["instrument"]["ffor"],settings["instrument"]["ffov"],settings["constellation"]["num_planes"],settings["agility"]["max_slew_rate"],
            settings["events"]["event_duration"],settings["events"]["event_frequency"],settings["events"]["event_density"],settings["events"]["event_clustering"],settings["num_meas_types"],
            settings["planner"],settings["rewards"]["reobserve_reward"], settings["rewards"]["reward"], overall_results["init_results"], elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()