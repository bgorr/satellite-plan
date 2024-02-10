import csv
import os
import time
import numpy as np
import shutil
import errno
import datetime

from run_kg_experiment import run_experiment

name = "kg_test"
kg_options = ["no_kg","det_kg","prob_kg"]
ukge_threshold_options = [0.25,0.5,0.75]
default_settings = {
        "name": "kg_test",
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
i = 0
settings_list = []
for option in kg_options:
    if option == "prob_kg":
        for threshold in ukge_threshold_options:
            experiment_name = 'kg_comparison_'+str(i)
            modified_settings = default_settings.copy()
            modified_settings["kg_setting"] = option
            modified_settings["ukge_threshold"] = threshold
            if modified_settings == default_settings:
                continue
            modified_settings["name"] = experiment_name
            modified_settings["directory"] = "./missions/"+experiment_name+"/"
            modified_settings["point_grid"] = "./coverage_grids/"+experiment_name+"/event_locations.csv"
            modified_settings["event_csvs"] = ["./events/"+experiment_name+"/events.csv"]
            settings_list.append(modified_settings)
            i = i+1
    else:
        experiment_name = 'kg_comparison_'+str(i)
        modified_settings = default_settings.copy()
        modified_settings["kg_setting"] = option
        modified_settings["ukge_threshold"] = "na"
        if modified_settings == default_settings:
            continue
        modified_settings["name"] = experiment_name
        modified_settings["directory"] = "./missions/"+experiment_name+"/"
        modified_settings["point_grid"] = "./coverage_grids/"+experiment_name+"/event_locations.csv"
        modified_settings["event_csvs"] = ["./events/"+experiment_name+"/events.csv"]
        settings_list.append(modified_settings)
        i = i+1
        
with open('./kg_comparison2.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
    first_row = ["name","for","fov","constellation_size","agility",
                "event_duration","event_frequency","event_density","event_clustering","num_meas_types","kg_option","ukge_threshold",
                "planner","reobserve_reward", "reward", "valid obs det_kg", "wasted effort det_kg", "valid obs ukge_0.25", "wasted effort ukge_0.5", "valid obs ukge_0.5", "wasted effort ukge_0.5","valid obs ukge_0.75", "wasted effort ukge_0.75", "num_vis", "time"]
    csvwriter.writerow(first_row)
    csvfile.close()
for settings in settings_list:
    start = time.time()
    print(settings["name"])
    # if settings["name"] != "kg_comparison_0":
    #     mission_src = "./missions/kg_comparison_0/"
    #     events_src = "./events/kg_comparison_0/"
    #     coverage_grids_src = "./coverage_grids/kg_comparison_0/"
    #     mission_dst = "./missions/"+settings["name"]+"/"
    #     events_dst = "./events/"+settings["name"]+"/"
    #     coverage_grids_dst = "./coverage_grids/"+settings["name"]+"/"
    #     try:
    #         # if not os.path.exists(mission_dst):
    #         #     os.makedirs(mission_dst)
    #         # if not os.path.exists(events_dst):
    #         #     os.makedirs(events_dst)
    #         # if not os.path.exists(coverage_grids_dst):
    #         #     os.makedirs(coverage_grids_dst)
    #         shutil.copytree(mission_src, mission_dst)
    #         shutil.copytree(events_src, events_dst)
    #         shutil.copytree(coverage_grids_src, coverage_grids_dst)
    #     except OSError as exc: # python >2.5
    #         if exc.errno in (errno.ENOTDIR, errno.EINVAL):
    #             shutil.copy(mission_src, mission_dst)
    #             shutil.copy(events_src, events_dst)
    #             shutil.copy(coverage_grids_src, coverage_grids_dst)
    #         else: raise
    overall_results = run_experiment(settings)
    end = time.time()
    elapsed_time = end-start
    with open('./kg_comparison2.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        row = [settings["name"],settings["instrument"]["ffor"],settings["instrument"]["ffov"],settings["constellation"]["num_planes"],settings["agility"]["max_slew_rate"],
            settings["events"]["event_duration"],settings["events"]["event_frequency"],settings["events"]["event_density"],settings["events"]["event_clustering"],settings["num_meas_types"],settings["kg_setting"],settings["ukge_threshold"],
            settings["planner"],settings["rewards"]["reobserve_reward"], settings["rewards"]["reward"], overall_results["num_obs_det"], overall_results["wasted_eff_det"], overall_results["num_obs_0.25"], overall_results["wasted_eff_0.25"],overall_results["num_obs_0.5"], overall_results["wasted_eff_0.5"],overall_results["num_obs_0.75"], overall_results["wasted_eff_0.75"], overall_results["num_vis"], elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()