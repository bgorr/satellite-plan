import csv
import os
import time
import numpy as np
import shutil
import errno

from run_experiment_het import run_experiment_het

experiment_num = 0
settings_list = []
event_durations = [2*3600,4*3600]
for event_duration in event_durations:
    settings = {
        "name": "grid_search_010624_"+str(experiment_num),
        "ffor": 30,
        "ffov": 0,
        "constellation_size": 6,
        "agility": 0.1,
        "event_duration": event_duration,
        "event_frequency": 0.1/3600,
        "event_density": 5,
        "event_clustering": 4,
        "planner": "dp",
        "reobserve_reward": 2,
        "num_meas_types": 2,
        "reward": 10,
        "reward_increment": 0.1,
        "sharing_horizon": 1000,
        "planning_horizon": 1000,
        "conops": "onboard_processing"
    }
    settings_list.append(settings)
    experiment_num += 1

        

with open('./grid_search_010624.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
    first_row = ["name","for","fov","constellation_size","agility",
                "event_duration","event_frequency","event_density","event_clustering","num_meas_types",
                "planner","reobserve_reward", "reward", "reward_increment", "sharing_horizon", "planning_horizon",
                "events",
                "init_event_obs_count","init_events_seen","init_event_reward","init_planner_reward","init_perc_cov","init_max_rev","init_avg_rev","init_all_perc_cov","init_all_max_rev","init_all_avg_rev",
                "replan_event_obs_count","replan_events_seen","replan_event_reward","replan_planner_reward","replan_perc_cov","replan_max_rev","replan_avg_rev","replan_all_perc_cov","replan_all_max_rev","replan_all_avg_rev",
                "replan_het_event_obs_count","replan_het_events_seen","replan_het_event_reward","replan_het_planner_reward","replan_het_perc_cov","replan_het_max_rev","replan_het_avg_rev","replan_het_all_perc_cov","replan_het_all_max_rev","replan_het_all_avg_rev",
                "vis_event_obs_count","vis_events_seen","vis_event_reward","vis_planner_reward","vis_perc_cov","vis_max_rev","vis_avg_rev","vis_all_perc_cov","vis_all_max_rev","vis_all_avg_rev",
                "time"]
    csvwriter.writerow(first_row)
    csvfile.close()
for settings in settings_list:
    print(settings["name"])
    start = time.time()
    overall_results = run_experiment_het(settings)
    end = time.time()
    elapsed_time = end-start
    with open('./grid_search_122123.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        row = [settings["name"],settings["instrument"]["ffor"],settings["instrument"]["ffov"],settings["constellation"]["num_planes"],settings["agility"]["max_slew_rate"],
            settings["events"]["event_duration"],settings["events"]["event_frequency"],settings["events"]["event_density"],settings["events"]["event_clustering"],settings["num_meas_types"],
            settings["planner"],settings["rewards"]["reobserve_reward"], settings["rewards"]["reward"], settings["rewards"]["reward_increment"], settings["sharing_horizon"], settings["planning_horizon"], 
            overall_results["num_events"],
            overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
            overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
            overall_results["replan_het_results"]["event_obs_count"],overall_results["replan_het_results"]["events_seen_once"],overall_results["replan_het_results"]["event_reward"],overall_results["replan_het_results"]["planner_reward"],overall_results["replan_het_results"]["percent_coverage"],overall_results["replan_het_results"]["event_max_revisit_time"],overall_results["replan_het_results"]["event_avg_revisit_time"],overall_results["replan_het_results"]["all_percent_coverage"],overall_results["replan_het_results"]["all_max_revisit_time"],overall_results["replan_het_results"]["all_avg_revisit_time"],
            overall_results["vis_results"]["event_obs_count"],overall_results["vis_results"]["events_seen_once"],overall_results["vis_results"]["event_reward"],overall_results["vis_results"]["planner_reward"],overall_results["vis_results"]["percent_coverage"],overall_results["vis_results"]["event_max_revisit_time"],overall_results["vis_results"]["event_avg_revisit_time"],overall_results["vis_results"]["all_percent_coverage"],overall_results["vis_results"]["all_max_revisit_time"],overall_results["vis_results"]["all_avg_revisit_time"],
            elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()