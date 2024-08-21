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
from src.plan_mission_fov import plan_mission_horizon, plan_mission_replan_interval, plan_mission_replan_oracle
from src.utils.compute_experiment_statistics import compute_experiment_statistics

def run_experiment(settings):
    simulation_step_size = settings["time"]["step_size"] # seconds
    simulation_duration = settings["time"]["duration"] # days
    mission_name = settings["name"]
    event_duration = settings["events"]["event_duration"]
    if not os.path.exists("./coverage_grids/"+mission_name+"/event_locations.csv"):
        possible_event_locations = []
        if settings["events"]["event_clustering"] == "uniform":
            center_lats = np.arange(-90,90,0.1)
            center_lons = np.arange(-180,180,0.1)
            for clat in center_lats:
                for clon in center_lons:
                    location = [clat,clon]
                    possible_event_locations.append(location)
        elif settings["events"]["event_clustering"] == "clustered":
            center_lats = np.random.uniform(-90,90,100)
            center_lons = np.random.uniform(-180,180,100)
            for i in range(len(center_lons)):
                var = 1
                mean = [center_lats[i], center_lons[i]]
                cov = [[var, 0], [0, var]]
                num_points_per_cell = int(6.48e6/100)
                xs, ys = np.random.multivariate_normal(mean, cov, num_points_per_cell).T
                for i in range(len(xs)):
                    location = [xs[i],ys[i]]
                    possible_event_locations.append(location)
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists("./missions/"+mission_name+"/events/"):
        events = []
        used_event_locations = []
        for i in range(settings["events"]["num_events"]):
            event_location = random.choice(possible_event_locations)
            used_event_locations.append(event_location)
            step = int(np.random.uniform(0,simulation_duration*86400))
            event = [event_location[0],event_location[1],step,event_duration,1]
            events.append(event)
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
    # plan_mission_replan_oracle(settings)
    overall_results = compute_experiment_statistics(settings)
    return overall_results


if __name__ == "__main__":
    with open('./updated_experiment_1.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        first_row = ["name","for","fov","num_planes","num_sats_per_plane","agility",
                    "event_duration","num_events","event_clustering","num_meas_types",
                    "planner","reobserve_reward", "reward",
                    "events","init_obs_count","replan_obs_count","oracle_obs_count",
                    "init_event_obs_count","init_events_seen_1+","init_events_seen_1","init_events_seen_2","init_events_seen_3","init_events_seen_4","init_event_reward","init_planner_reward","init_perc_cov","init_max_rev","init_avg_rev","init_all_perc_cov","init_all_max_rev","init_all_avg_rev",
                    "replan_event_obs_count","replan_events_seen_1+","replan_events_seen_1","replan_events_seen_2","replan_events_seen_3","replan_events_seen_4+","replan_event_reward","replan_planner_reward","replan_perc_cov","replan_max_rev","replan_avg_rev","replan_all_perc_cov","replan_all_max_rev","replan_all_avg_rev",
                    "oracle_event_obs_count","oracle_events_seen_1+","oracle_events_seen_1","oracle_events_seen_2","oracle_events_seen_3","oracle_events_seen_4+","oracle_event_reward","oracle_planner_reward","oracle_perc_cov","oracle_max_rev","oracle_avg_rev","oracle_all_perc_cov","oracle_all_max_rev","oracle_all_avg_rev",
                    "time"]
        csvwriter.writerow(first_row)
        csvfile.close()
    if os.path.exists("./resources/lhs_49.txt"):
        scaled_sample = np.loadtxt("./resources/lhs_49.txt",dtype=float)
    else:
        sampler = qmc.LatinHypercube(d=7, strength=2, optimization="lloyd")

        sample = sampler.random(n=49)
        scaled_sample = qmc.scale(sample,[0,0,0,0,0,0,0],[4,2,3,3,2,2,4])
        scaled_sample_save = np.array(scaled_sample)
        np.savetxt('lhs_49.txt', scaled_sample_save, fmt='%1.3f')

    constellation_options = [(2,2),(1,4),(3,8),(8,3)]
    for_options = [30,60]
    fov_options = [1,5,10]
    agility_options = [0.1,1,10]
    event_clustering_options = ["uniform","clustered"]
    event_quantity_options = [1000,10000]
    event_duration_options = [15*60,3600,3*3600,6*3600]
    i = 0
    start_ind = 0
    for i in range(start_ind,len(scaled_sample)):
        design = scaled_sample[i]
        constellation_option = constellation_options[int(np.floor(design[0]))]
        for_option = for_options[int(np.floor(design[1]))]
        fov_option = fov_options[int(np.floor(design[2]))]
        agility_option = agility_options[int(np.floor(design[3]))]
        event_clustering_option = event_clustering_options[int(np.floor(design[4]))]
        event_quantity_option = event_quantity_options[int(np.floor(design[5]))]
        event_duration_option = event_duration_options[int(np.floor(design[6]))]
        name = "updated_experiment_1_"+str(i)
        settings = {
            "name": name,
            "instrument": {
                "ffor": for_option,
                "ffov": fov_option
            },
            "agility": {
                "slew_constraint": "rate",
                "max_slew_rate": agility_option,
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
                "event_duration": event_duration_option,
                "num_events": int(event_quantity_option),
                "event_clustering": event_clustering_option
            },
            "time": {
                "step_size": 10, # seconds
                "duration": 1, # days
                "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
            },
            "rewards": {
                "reward": 10,
                "reward_increment": 0.1,
                "reobserve_reward": 2,
                "reobserve_conops": "linear_increase",
                "event_duration_decay": "linear",
                "no_event_reward": 0,
                "oracle_reobs": "true",
                "initial_reward": 1
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
        with open('./updated_experiment_1.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
            row = [settings["name"],settings["instrument"]["ffor"],settings["instrument"]["ffov"],settings["constellation"]["num_planes"],settings["constellation"]["num_sats_per_plane"],settings["agility"]["max_slew_rate"],
                    settings["events"]["event_duration"],settings["events"]["num_events"],settings["events"]["event_clustering"],settings["num_meas_types"],
                    settings["planner"],settings["sharing_horizon"], settings["planning_horizon"], settings["rewards"]["reward"], settings["rewards"]["reward_increment"], settings["rewards"]["reobserve_conops"],settings["rewards"]["event_duration_decay"],settings["rewards"]["no_event_reward"],
                overall_results["num_events"],overall_results["num_obs_init"],overall_results["num_obs_replan"],overall_results["num_obs_oracle"],
                overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["events_seen_at_least_once"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["events_seen_twice"],overall_results["init_results"]["events_seen_thrice"],overall_results["init_results"]["events_seen_fourplus"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
                overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["events_seen_at_least_once"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["events_seen_twice"],overall_results["replan_results"]["events_seen_thrice"],overall_results["replan_results"]["events_seen_fourplus"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
                overall_results["oracle_results"]["event_obs_count"],overall_results["oracle_results"]["events_seen_at_least_once"],overall_results["oracle_results"]["events_seen_once"],overall_results["oracle_results"]["events_seen_twice"],overall_results["oracle_results"]["events_seen_thrice"],overall_results["oracle_results"]["events_seen_fourplus"],overall_results["oracle_results"]["event_reward"],overall_results["oracle_results"]["planner_reward"],overall_results["oracle_results"]["percent_coverage"],overall_results["oracle_results"]["event_max_revisit_time"],overall_results["oracle_results"]["event_avg_revisit_time"],overall_results["oracle_results"]["all_percent_coverage"],overall_results["oracle_results"]["all_max_revisit_time"],overall_results["oracle_results"]["all_avg_revisit_time"],
                elapsed_time
            ]
            csvwriter.writerow(row)
            csvfile.close()