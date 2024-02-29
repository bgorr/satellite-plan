import datetime
import os
import numpy as np
import csv
import time
import shutil, errno
import random
import tqdm
from scipy.stats import qmc
import sys
sys.path.append(".")

from src.create_mission import create_mission
from src.execute_mission import execute_mission
from src.plan_mission_fov import plan_mission_horizon, plan_mission_replan_interval
from src.utils.compute_experiment_statistics import compute_experiment_statistics

def run_experiment(settings):
    mission_name = settings["name"]
    event_csv = './src/utils/event_csvs/lightning_events.csv'
    events = []
    used_event_locations = []
    event_durations = []
    with open(event_csv,'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            event_location = [float(row[0]),float(row[1])]
            if event_location not in used_event_locations:
                used_event_locations.append(event_location)
            event = [event_location[0],event_location[1],float(row[2]),float(row[3]),1]
            event_durations.append(float(row[3]))
            events.append(event)
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists("./missions/"+mission_name+"/events/"):
        if not os.path.exists("./missions/"+mission_name+"/events/"):
            os.mkdir("./missions/"+mission_name+"/events/")
        with open("./missions/"+mission_name+"/events/events.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
            for event in events:
                csvwriter.writerow(event)
    if not os.path.exists("./missions/"+mission_name+"/coverage_grids/"):
        os.mkdir("./missions/"+mission_name+"/coverage_grids/")
        with open("./missions/"+mission_name+"/coverage_grids/event_locations.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]'])
            for location in used_event_locations:
                csvwriter.writerow(location)
    

    
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
        create_mission(settings)
        execute_mission(settings)
    plan_mission_horizon(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    plan_mission_replan_interval(settings)
    overall_results = compute_experiment_statistics(settings)
    return overall_results


if __name__ == "__main__":
    with open('./studies/lightning_constellation_study.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        first_row = ["name","for","fov","num_planes","num_sats_per_plane","agility",
                    "event_duration","num_events","event_clustering","num_meas_types",
                    "planner","sharing_horizon", "planning_horizon", "reward", "reward_increment", "reobserve_conops","event_duration_decay","no_event_reward",
                    "events","init_obs_count","replan_obs_count","oracle_obs_count",
                    "init_event_obs_count","init_obs_per_event_list","init_events_seen_1+","init_events_seen_1","init_events_seen_2","init_events_seen_3","init_events_seen_4","init_event_reward","init_planner_reward","init_perc_cov","init_max_rev","init_avg_rev","init_all_perc_cov","init_all_max_rev","init_all_avg_rev",
                    "replan_event_obs_count","replan_obs_per_event_list","replan_events_seen_1+","replan_events_seen_1","replan_events_seen_2","replan_events_seen_3","replan_events_seen_4+","replan_event_reward","replan_planner_reward","replan_perc_cov","replan_max_rev","replan_avg_rev","replan_all_perc_cov","replan_all_max_rev","replan_all_avg_rev",
                    "time"]
        csvwriter.writerow(first_row)
        csvfile.close()

    constellation_options = [(2,2),(1,4),(3,8),(8,3)]
    for_options = [30,60]
    fov_options = [1,5,10]
    agility_options = [0.1,1,10]
    event_clustering_options = ["uniform","clustered"]
    event_quantity_options = [1000,10000]
    event_duration_options = [15*60,3600,3*3600,6*3600]

    event_csv = './src/utils/event_csvs/lightning_events.csv'
    event_durations = []
    num_events = 0
    with open(event_csv,'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            event_durations.append(float(row[3]))
            num_events += 1
    average_event_duration = np.mean(event_durations)

    i = 0
    for constellation_option in constellation_options:
        name = "lightning_constellation_study_"+str(i)
        settings = {
            "name": name,
            "instrument": {
                "ffor": 60,
                "ffov": 5
            },
            "agility": {
                "slew_constraint": "rate",
                "max_slew_rate": 1,
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
                "num_sats_per_plane": constellation_option[1],
                "num_planes": constellation_option[0],
                "phasing_parameter": 1
            },
            "events": {
                "event_duration": average_event_duration,
                "num_events": num_events,
                "event_clustering": "clustered"
            },
            "time": {
                "step_size": 10, # seconds
                "duration": 1, # days
                "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
            },
            "rewards": {
                "reward": 10,
                "reward_increment": 1,
                "reobserve_conops": "no_change",
                "event_duration_decay": "step",
                "no_event_reward": 5,
                "oracle_reobs": "true",
                "initial_reward": 5
            },
            "planner": "dp",
            "num_meas_types": 3,
            "sharing_horizon": 500,
            "planning_horizon": 500,
            "directory": "./missions/"+name+"/",
            "grid_type": "custom", # can be "uniform" or "custom"
            "point_grid": "./missions/"+name+"/coverage_grids/event_locations.csv",
            "preplanned_observations": None,
            "event_csvs": ["./missions/"+name+"/events/events.csv"],
            "process_obs_only": False,
            "conops": "onboard_processing"
        }
        start = time.time()
        print(settings["name"])
        overall_results = run_experiment(settings)
        end = time.time()
        elapsed_time = end-start
        with open('./studies/lightning_constellation_study.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
            row = [settings["name"],settings["instrument"]["ffor"],settings["instrument"]["ffov"],settings["constellation"]["num_planes"],settings["constellation"]["num_sats_per_plane"],settings["agility"]["max_slew_rate"],
                settings["events"]["event_duration"],settings["events"]["num_events"],settings["events"]["event_clustering"],settings["num_meas_types"],
                settings["planner"],settings["sharing_horizon"], settings["planning_horizon"], settings["rewards"]["reward"], settings["rewards"]["reward_increment"], settings["rewards"]["reobserve_conops"],settings["rewards"]["event_duration_decay"],settings["rewards"]["no_event_reward"],
                overall_results["num_events"],overall_results["num_obs_init"],overall_results["num_obs_replan"],overall_results["num_obs_oracle"],
                overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["obs_per_event_list"],overall_results["init_results"]["events_seen_at_least_once"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["events_seen_twice"],overall_results["init_results"]["events_seen_thrice"],overall_results["init_results"]["events_seen_fourplus"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
                overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["obs_per_event_list"],overall_results["replan_results"]["events_seen_at_least_once"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["events_seen_twice"],overall_results["replan_results"]["events_seen_thrice"],overall_results["replan_results"]["events_seen_fourplus"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
                elapsed_time
            ]
            csvwriter.writerow(row)
            csvfile.close()
        i = i+1