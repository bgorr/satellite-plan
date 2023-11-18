import csv
import os
import time
import numpy as np
import shutil
import errno

from run_experiment_het import run_experiment_het

experiment_num = 0
settings_list = []
rewards = [5,10,50]
reward_increments = [0.1,0.5,1.0]
time_horizons = [500,1000,5000]
for reward in rewards:
    for reward_increment in reward_increments:
        for time_horizon in time_horizons:
            settings = {
                "name": "grid_search_"+str(experiment_num),
                "ffor": 30,
                "ffov": 0,
                "constellation_size": 2,
                "agility": 0.1,
                "event_duration": 3600*4,
                "event_frequency": 0.1/3600,
                "event_density": 5,
                "event_clustering": 1,
                "planner": "dp",
                "reobserve_reward": 2,
                "num_meas_types": 3,
                "reward": reward,
                "reward_increment": reward_increment,
                "time_horizon": time_horizon
            }
            settings_list.append(settings)
            experiment_num += 1

        

with open('./grid_search_111723.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
    first_row = ["name","for","fov","constellation_size","agility",
                "event_duration","event_frequency","event_density","event_clustering","num_meas_types",
                "planner","reobserve_reward", "reward", "reward_increment", "time_horizon",
                "events","replan_obs_count","replan_het_obs_count",
                #"init_event_obs_count","init_events_seen","init_event_reward","init_planner_reward","init_perc_cov","init_max_rev","init_avg_rev","init_all_perc_cov","init_all_max_rev","init_all_avg_rev",
                "replan_event_obs_count","replan_events_seen","replan_event_reward","replan_planner_reward","replan_perc_cov","replan_max_rev","replan_avg_rev","replan_all_perc_cov","replan_all_max_rev","replan_all_avg_rev",
                #"replan_het_event_obs_count","replan_het_events_seen","replan_het_event_reward","replan_het_planner_reward","replan_het_perc_cov","replan_het_max_rev","replan_het_avg_rev","replan_het_all_perc_cov","replan_het_all_max_rev","replan_het_all_avg_rev",
                "vis_event_obs_count","vis_events_seen","vis_event_reward","vis_planner_reward","vis_perc_cov","vis_max_rev","vis_avg_rev","vis_all_perc_cov","vis_all_max_rev","vis_all_avg_rev",
                "time"]
    csvwriter.writerow(first_row)
    csvfile.close()
for settings in settings_list:
    print(settings["name"])
    if settings["name"] != "grid_search_0":
        mission_src = "./missions/grid_search_0/"
        events_src = "./events/grid_search_0/"
        coverage_grids_src = "./coverage_grids/grid_search_0/"
        mission_dst = "./missions/"+settings["name"]+"/"
        events_dst = "./events/"+settings["name"]+"/"
        coverage_grids_dst = "./coverage_grids/"+settings["name"]+"/"
        try:
            # if not os.path.exists(mission_dst):
            #     os.makedirs(mission_dst)
            # if not os.path.exists(events_dst):
            #     os.makedirs(events_dst)
            # if not os.path.exists(coverage_grids_dst):
            #     os.makedirs(coverage_grids_dst)
            shutil.copytree(mission_src, mission_dst)
            shutil.copytree(events_src, events_dst)
            shutil.copytree(coverage_grids_src, coverage_grids_dst)
        except OSError as exc: # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(mission_src, mission_dst)
                shutil.copy(events_src, events_dst)
                shutil.copy(coverage_grids_src, coverage_grids_dst)
            else: raise
    start = time.time()
    overall_results = run_experiment_het(settings)
    end = time.time()
    elapsed_time = end-start
    with open('./grid_search_111723.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        row = [settings["name"],settings["ffor"],settings["ffov"],settings["constellation_size"],settings["agility"],
            settings["event_duration"],settings["event_frequency"],settings["event_density"],settings["event_clustering"],settings["num_meas_types"],
            settings["planner"],settings["reobserve_reward"], settings["reward"], settings["reward_increment"], settings["time_horizon"],
            overall_results["num_events"],overall_results["num_obs_replan"],overall_results["num_obs_replan_het"],
            #overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
            overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
            overall_results["replan_het_results"]["event_obs_count"],overall_results["replan_het_results"]["events_seen_once"],overall_results["replan_het_results"]["event_reward"],overall_results["replan_het_results"]["planner_reward"],overall_results["replan_het_results"]["percent_coverage"],overall_results["replan_het_results"]["event_max_revisit_time"],overall_results["replan_het_results"]["event_avg_revisit_time"],overall_results["replan_het_results"]["all_percent_coverage"],overall_results["replan_het_results"]["all_max_revisit_time"],overall_results["replan_het_results"]["all_avg_revisit_time"],
            #overall_results["vis_results"]["event_obs_count"],overall_results["vis_results"]["events_seen_once"],overall_results["vis_results"]["event_reward"],overall_results["vis_results"]["planner_reward"],overall_results["vis_results"]["percent_coverage"],overall_results["vis_results"]["event_max_revisit_time"],overall_results["vis_results"]["event_avg_revisit_time"],overall_results["vis_results"]["all_percent_coverage"],overall_results["vis_results"]["all_max_revisit_time"],overall_results["vis_results"]["all_avg_revisit_time"],
            elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()