import datetime
import os
import numpy as np
import csv
import time

from create_mission import create_mission
from execute_mission import execute_mission
from plan_mission import plan_mission, plan_mission_replan_interval, plan_mission_replan_interval_het
from utils.compute_experiment_statistics_het import compute_experiment_statistics_het

def run_experiment_het(settings):
    simulation_duration = settings["time"]["duration"]
    simulation_step_size= settings["time"]["step_size"]
    mission_name = settings["name"]
    var = settings["events"]["event_clustering"]
    num_points_per_cell = settings["events"]["event_density"]
    event_frequency = settings["events"]["event_frequency"]
    event_duration = settings["events"]["event_duration"]
    num_meas_types = settings["num_meas_types"]
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
        for step in steps:
            for location in event_locations:
                if np.random.random() < event_frequency*simulation_step_size:
                    if num_meas_types == 2:
                        meas_types_needed = [0,1]
                    elif num_meas_types == 3:
                        meas_types_needed = [0,1,2]
                    elif num_meas_types == 4:
                        meas_types_needed = [0,1,2,3]
                    event = [location[0],location[1],step,event_duration,1,meas_types_needed]
                    events.append(event)
        if not os.path.exists("./events/"+mission_name+"/"):
            os.mkdir("./events/"+mission_name+"/")
        with open("./events/"+mission_name+"/events.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity','meas_types_needed'])
            for event in events:
                csvwriter.writerow(event)

    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
        create_mission(settings)
        execute_mission(settings)
    plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    plan_mission_replan_interval(settings)
    plan_mission_replan_interval_het(settings)
    overall_results = compute_experiment_statistics_het(settings)
    return overall_results


if __name__ == "__main__":

    with open('./test_het.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        first_row = ["name","for","fov","constellation_size","agility",
                    "event_duration","event_frequency","event_density","event_clustering","num_meas_types",
                    "planner","reobserve_reward", "reward",
                    "events","init_obs_count","replan_obs_count","replan_het_obs_count","vis_count",
                    "init_event_obs_count","init_events_seen","init_event_reward","init_planner_reward","init_perc_cov","init_max_rev","init_avg_rev","init_all_perc_cov","init_all_max_rev","init_all_avg_rev",
                    "replan_event_obs_count","replan_events_seen","replan_event_reward","replan_planner_reward","replan_perc_cov","replan_max_rev","replan_avg_rev","replan_all_perc_cov","replan_all_max_rev","replan_all_avg_rev",
                    "replan_het_event_obs_count","replan_het_events_seen","replan_het_event_reward","replan_het_planner_reward","replan_het_perc_cov","replan_het_max_rev","replan_het_avg_rev","replan_het_all_perc_cov","replan_het_all_max_rev","replan_het_all_avg_rev",
                    "vis_event_obs_count","vis_events_seen","vis_event_reward","vis_planner_reward","vis_perc_cov","vis_max_rev","vis_avg_rev","vis_all_perc_cov","vis_all_max_rev","vis_all_avg_rev",
                    "time"]
        csvwriter.writerow(first_row)
        csvfile.close()

    name = "test_het"
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
            "duration": 0.1, # days
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
    overall_results = run_experiment_het(settings)
    end = time.time()
    elapsed_time = end-start
    with open('./test_het.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        row = [settings["name"],settings["instrument"]["ffor"],settings["instrument"]["ffov"],settings["constellation"]["num_planes"],settings["agility"]["max_slew_rate"],
            settings["events"]["event_duration"],settings["events"]["event_frequency"],settings["events"]["event_density"],settings["events"]["event_clustering"],settings["num_meas_types"],
            settings["planner"],settings["rewards"]["reobserve_reward"], settings["rewards"]["reward"],
            overall_results["num_events"],overall_results["num_obs_init"],overall_results["num_obs_replan"],overall_results["num_obs_replan_het"],overall_results["num_vis"],
            overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
            overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
            overall_results["replan_het_results"]["event_obs_count"],overall_results["replan_het_results"]["events_seen_once"],overall_results["replan_het_results"]["event_reward"],overall_results["replan_het_results"]["planner_reward"],overall_results["replan_het_results"]["percent_coverage"],overall_results["replan_het_results"]["event_max_revisit_time"],overall_results["replan_het_results"]["event_avg_revisit_time"],overall_results["replan_het_results"]["all_percent_coverage"],overall_results["replan_het_results"]["all_max_revisit_time"],overall_results["replan_het_results"]["all_avg_revisit_time"],
            overall_results["vis_results"]["event_obs_count"],overall_results["vis_results"]["events_seen_once"],overall_results["vis_results"]["event_reward"],overall_results["vis_results"]["planner_reward"],overall_results["vis_results"]["percent_coverage"],overall_results["vis_results"]["event_max_revisit_time"],overall_results["vis_results"]["event_avg_revisit_time"],overall_results["vis_results"]["all_percent_coverage"],overall_results["vis_results"]["all_max_revisit_time"],overall_results["vis_results"]["all_avg_revisit_time"],
            elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()