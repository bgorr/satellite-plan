import csv
import os
import time
import oapackage
import oapackage.Doptim
import numpy as np

from run_experiment_het import run_experiment_het

event_frequency_levels = [0.1/3600,0.01/3600,0.001/3600,1e-4/3600]
agility_levels = [5,1,0.1,0.01]
ffor_levels = [10,30,60]
ffov_levels = [10,5,1]
event_duration_levels = [12*3600,6*3600,3600]
event_density_levels = [1,5,10]
event_clustering_levels = [1,4,8]
event_type_levels = [2,3,4]
constellation_size_levels = [2,3]

run_size = 200
number_of_factors = 8
factor_levels = [4,4,3,3,3,3,3,3]
strength = 0


arrayclass = oapackage.arraydata_t(factor_levels, run_size, strength, number_of_factors)

alpha=[1,2,0]

scores, design_efficiencies, designs, ngenerated = oapackage.Doptim.Doptimize(
    arrayclass, nrestarts=40, optimfunc=alpha, selectpareto=True
)
print('Generated %d designs, the best D-efficiency is %.4f' % (len(designs), design_efficiencies[:,0].max() ))
selected_array = designs[0]
print("The array is (in transposed form):\n")
selected_array.transposed().showarraycompact()
oa = np.array(selected_array)
print(oa)

experiment_num = 6
settings_list = []
for i in range(6,len(oa[:,0])):
    settings = {
        "name": "oa_het_"+str(experiment_num),
        "ffor": ffor_levels[oa[i,2]],
        "ffov": ffov_levels[oa[i,3]],
        "constellation_size": constellation_size_levels[0],
        "agility": agility_levels[oa[i,1]],
        "event_duration": event_duration_levels[oa[i,4]],
        "event_frequency": event_frequency_levels[oa[i,0]],
        "event_density": event_density_levels[oa[i,5]],
        "event_clustering": event_clustering_levels[oa[i,6]],
        "planner": "dp",
        "reobserve_reward": 2,
        "num_event_types": event_type_levels[oa[i,7]],
        "reward": 10
    }
    settings_list.append(settings)
    experiment_num += 1

        

with open('./oa_results_het_101223.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
    first_row = ["name","for","fov","constellation_size","agility",
                "event_duration","event_frequency","event_density","event_clustering","num_event_types",
                "planner","reobserve_reward", "reward",
                "events","init_obs_count","replan_obs_count","vis_count",
                "init_event_obs_count","init_events_seen","init_event_reward","init_planner_reward","init_perc_cov","init_max_rev","init_avg_rev","init_all_perc_cov","init_all_max_rev","init_all_avg_rev",
                "replan_event_obs_count","replan_events_seen","replan_event_reward","replan_planner_reward","replan_perc_cov","replan_max_rev","replan_avg_rev","replan_all_perc_cov","replan_all_max_rev","replan_all_avg_rev",
                "vis_event_obs_count","vis_events_seen","vis_event_reward","vis_planner_reward","vis_perc_cov","vis_max_rev","vis_avg_rev","vis_all_perc_cov","vis_all_max_rev","vis_all_avg_rev",
                "time"]
    csvwriter.writerow(first_row)
    csvfile.close()
for settings in settings_list:
    start = time.time()
    print(settings["name"])
    overall_results = run_experiment_het(settings)
    end = time.time()
    elapsed_time = end-start
    with open('./oa_results_het_101223.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        row = [settings["name"],settings["ffor"],settings["ffov"],settings["constellation_size"],settings["agility"],
            settings["event_duration"],settings["event_frequency"],settings["event_density"],settings["event_clustering"],settings["num_event_types"],
            settings["planner"],settings["reobserve_reward"], settings["reward"],
            overall_results["num_events"],overall_results["num_obs_init"],overall_results["num_obs_replan"],overall_results["num_vis"],
            overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
            overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
            overall_results["vis_results"]["event_obs_count"],overall_results["vis_results"]["events_seen_once"],overall_results["vis_results"]["event_reward"],overall_results["vis_results"]["planner_reward"],overall_results["vis_results"]["percent_coverage"],overall_results["vis_results"]["event_max_revisit_time"],overall_results["vis_results"]["event_avg_revisit_time"],overall_results["vis_results"]["all_percent_coverage"],overall_results["vis_results"]["all_max_revisit_time"],overall_results["vis_results"]["all_avg_revisit_time"],
            elapsed_time
        ]
        csvwriter.writerow(row)
        csvfile.close()