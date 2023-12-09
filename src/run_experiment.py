import datetime
import os
import numpy as np
import csv
import time
import shutil, errno

from create_mission import create_mission
from execute_mission import execute_mission
from process_mission import process_mission
from plan_mission import plan_mission, plan_mission_replan_interval
from plot_mission_cartopy import plot_mission
from utils.compute_experiment_statistics import compute_experiment_statistics

def run_experiment(experiment_settings):
    # mission_name = "experiment2"
    # cross_track_ffor = 60 # deg
    # along_track_ffor = 60 # deg
    # cross_track_ffov = 10 # deg
    # along_track_ffov = 10 # deg
    # agility = 1 # deg/s
    # num_planes = 4
    # num_sats_per_plane = 4
    # var = 5 # deg lat/lon
    # num_points_per_cell = 10
    simulation_step_size = 10 # seconds
    simulation_duration = 1 # days
    # event_frequency = 0.001/3600 # events per hour
    # event_duration = 3600*6 # second
    mission_name = experiment_settings["name"]
    cross_track_ffor = experiment_settings["ffor"]
    along_track_ffor = experiment_settings["ffor"]
    cross_track_ffov = experiment_settings["ffov"]
    along_track_ffov = experiment_settings["ffov"] # TODO carefully consider this assumption
    agility = experiment_settings["agility"]
    num_planes = experiment_settings["constellation_size"]
    num_sats_per_plane = experiment_settings["constellation_size"]
    var = experiment_settings["event_clustering"]
    num_points_per_cell = experiment_settings["event_density"]
    event_frequency = experiment_settings["event_frequency"]
    event_duration = experiment_settings["event_duration"]
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
                    event = [location[0],location[1],step,event_duration,1]
                    events.append(event)
        if not os.path.exists("./events/"+mission_name+"/"):
            os.mkdir("./events/"+mission_name+"/")
        with open("./events/"+mission_name+"/events.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
            for event in events:
                csvwriter.writerow(event)

    settings = {
        "directory": "./missions/"+mission_name+"/",
        "step_size": simulation_step_size,
        "duration": simulation_duration,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "point_grid": "./coverage_grids/"+mission_name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+mission_name+"/events.csv"],
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False,
        "planner": experiment_settings["planner"],
        "reward": experiment_settings["reward"],
        "reward_increment": experiment_settings["reward_increment"],
        "sharing_horizon": experiment_settings["sharing_horizon"],
        "planning_horizon": experiment_settings["planning_horizon"],
        "reobserve_reward": experiment_settings["reobserve_reward"],
        "reward": experiment_settings["reward"],
        "experiment_settings": experiment_settings
    }
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
        
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
        create_mission(settings)
        execute_mission(settings)
    plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    plan_mission_replan_interval(settings)
    overall_results = compute_experiment_statistics(settings)
    return overall_results


if __name__ == "__main__":

    with open('./milp_test_hom.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        first_row = ["name","for","fov","constellation_size","agility",
                    "event_duration","event_frequency","event_density","event_clustering","num_meas_types",
                    "planner","reobserve_reward", "reward",
                    "events","init_obs_count","replan_obs_count",#"replan_het_obs_count","vis_count",
                    "init_event_obs_count","init_events_seen","init_event_reward","init_planner_reward","init_perc_cov","init_max_rev","init_avg_rev","init_all_perc_cov","init_all_max_rev","init_all_avg_rev",
                    "replan_event_obs_count","replan_events_seen","replan_event_reward","replan_planner_reward","replan_perc_cov","replan_max_rev","replan_avg_rev","replan_all_perc_cov","replan_all_max_rev","replan_all_avg_rev",
                    #"replan_het_event_obs_count","replan_het_events_seen","replan_het_event_reward","replan_het_planner_reward","replan_het_perc_cov","replan_het_max_rev","replan_het_avg_rev","replan_het_all_perc_cov","replan_het_all_max_rev","replan_het_all_avg_rev",
                    "vis_event_obs_count","vis_events_seen","vis_event_reward","vis_planner_reward","vis_perc_cov","vis_max_rev","vis_avg_rev","vis_all_perc_cov","vis_all_max_rev","vis_all_avg_rev",
                    "time"]
        csvwriter.writerow(first_row)
        csvfile.close()

    settings = {
        "name": "milp_test_hom",
        "ffor": 10,
        "ffov": 0,
        "constellation_size": 6,
        "agility": 0.1,
        "event_duration": 3600*6,
        "event_frequency": 0.01/3600,
        "event_density": 2,
        "event_clustering": 4,
        "planner": "milp",
        "reobserve_reward": 2,
        "num_meas_types": 3,
        "reward": 10,
        "reward_increment": 0.1,
        "sharing_horizon": 1000,
        "planning_horizon": 1000
    }
    start = time.time()
    print(settings["name"])
    # if settings["name"] != "milp_test_hom":
    #     mission_src = "./missions/milp_test_hom/"
    #     events_src = "./events/milp_test_hom/"
    #     coverage_grids_src = "./coverage_grids/milp_test_hom/"
    #     mission_dst = "./missions/"+settings["name"]+"/"
    #     events_dst = "./events/"+settings["name"]+"/"
    #     coverage_grids_dst = "./coverage_grids/"+settings["name"]+"/"
    #     try:
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
    with open('./milp_test_hom.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        row = [settings["name"],settings["ffor"],settings["ffov"],settings["constellation_size"],settings["agility"],
            settings["event_duration"],settings["event_frequency"],settings["event_density"],settings["event_clustering"],settings["num_meas_types"],
            settings["planner"],settings["reobserve_reward"], settings["reward"],
            overall_results["num_events"],overall_results["num_obs_init"],overall_results["num_obs_replan"],
            overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
            overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
            #overall_results["replan_het_results"]["event_obs_count"],overall_results["replan_het_results"]["events_seen_once"],overall_results["replan_het_results"]["event_reward"],overall_results["replan_het_results"]["planner_reward"],overall_results["replan_het_results"]["percent_coverage"],overall_results["replan_het_results"]["event_max_revisit_time"],overall_results["replan_het_results"]["event_avg_revisit_time"],overall_results["replan_het_results"]["all_percent_coverage"],overall_results["replan_het_results"]["all_max_revisit_time"],overall_results["replan_het_results"]["all_avg_revisit_time"],
            overall_results["vis_results"]["event_obs_count"],overall_results["vis_results"]["events_seen_once"],overall_results["vis_results"]["event_reward"],overall_results["vis_results"]["planner_reward"],overall_results["vis_results"]["percent_coverage"],overall_results["vis_results"]["event_max_revisit_time"],overall_results["vis_results"]["event_avg_revisit_time"],overall_results["vis_results"]["all_percent_coverage"],overall_results["vis_results"]["all_max_revisit_time"],overall_results["vis_results"]["all_avg_revisit_time"],
            elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()