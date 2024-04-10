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
from flask import Flask, request

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
        
    if not os.path.exists("./events/"+mission_name+"/events.csv"):
        events = []
        used_event_locations = []
        for i in range(settings["events"]["num_events"]):
            event_location = random.choice(possible_event_locations)
            used_event_locations.append(event_location)
            step = int(np.random.uniform(0,simulation_duration*86400))
            event = [event_location[0],event_location[1],step,event_duration,1]
            events.append(event)
        if not os.path.exists("./events/"+mission_name+"/"):
            os.mkdir("./events/"+mission_name+"/")
        with open("./events/"+mission_name+"/events.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
            for event in events:
                csvwriter.writerow(event)
    if not os.path.exists("./coverage_grids/"+mission_name+"/"):
        os.mkdir("./coverage_grids/"+mission_name+"/")
        with open("./coverage_grids/"+mission_name+"/event_locations.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]'])
            for location in used_event_locations:
                csvwriter.writerow(location)
    

    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
        create_mission(settings)
        execute_mission(settings)
    #plan_mission_horizon(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    plan_mission_replan_interval(settings)
    overall_results = compute_experiment_statistics(settings)
    return overall_results


app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        print(request.values)
        n_s = int(request.form.get("n_s"))
        n_p = int(request.form.get("n_p"))
        alt = float(request.form.get("alt"))
        inc = float(request.form.get("inc"))
        fov = float(request.form.get("fov"))
        agility = float(request.form.get("agility"))
        name = "run_experiment_"+str(n_s)+str(n_p)+str(alt)+str(np.round(inc,2))+str(np.round(fov,2))+str(np.round(agility,2))
        settings = {
            "name": name,
            "instrument": {
                "ffor": 60,
                "ffov": fov
            },
            "agility": {
                "slew_constraint": "rate",
                "max_slew_rate": agility,
                "inertia": 2.66,
                "max_torque": 4e-3
            },
            "orbit": {
                "altitude": alt, # km
                "inclination": inc, # deg
                "eccentricity": 0.0001,
                "argper": 0, # deg
            },
            "constellation": {
                "num_sats_per_plane": n_s,
                "num_planes": n_p,
                "phasing_parameter": 1
            },
            "events": {
                "event_duration": 3600*4,
                "num_events": int(1000),
                "event_clustering": "uniform"
            },
            "time": {
                "step_size": 10, # seconds
                "duration": 1, # days
                "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
            },
            "rewards": {
            "reward": 10,
            "reward_increment": 1,
            "reobserve_conops": "linear_increase",
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
            "point_grid": "./coverage_grids/"+name+"/event_locations.csv",
            "preplanned_observations": None,
            "event_csvs": ["./events/"+name+"/events.csv"],
            "process_obs_only": False,
            "conops": "onboard_processing"
        }
        overall_results = run_experiment(settings)
        reward = overall_results["replan_results"]["event_obs_count"]
        # with open('./updated_experiment_test.csv','a') as csvfile:
        #     csvwriter = csv.writer(csvfile,delimiter=',',quotechar='|')
        #     row = [settings["name"],settings["instrument"]["ffor"],settings["instrument"]["ffov"],settings["constellation"]["num_planes"],settings["constellation"]["num_sats_per_plane"],settings["agility"]["max_slew_rate"],
        #         settings["events"]["event_duration"],settings["events"]["num_events"],settings["events"]["event_clustering"],settings["num_meas_types"],
        #         settings["planner"],settings["rewards"]["reobserve_reward"], settings["rewards"]["reward"],
        #         overall_results["num_events"],overall_results["num_obs_init"],overall_results["num_obs_replan"],overall_results["num_obs_oracle"],
        #         overall_results["init_results"]["event_obs_count"],overall_results["init_results"]["events_seen_at_least_once"],overall_results["init_results"]["events_seen_once"],overall_results["init_results"]["events_seen_twice"],overall_results["init_results"]["events_seen_thrice"],overall_results["init_results"]["events_seen_fourplus"],overall_results["init_results"]["event_reward"],overall_results["init_results"]["planner_reward"],overall_results["init_results"]["percent_coverage"],overall_results["init_results"]["event_max_revisit_time"],overall_results["init_results"]["event_avg_revisit_time"],overall_results["init_results"]["all_percent_coverage"],overall_results["init_results"]["all_max_revisit_time"],overall_results["init_results"]["all_avg_revisit_time"],
        #         overall_results["replan_results"]["event_obs_count"],overall_results["replan_results"]["events_seen_at_least_once"],overall_results["replan_results"]["events_seen_once"],overall_results["replan_results"]["events_seen_twice"],overall_results["replan_results"]["events_seen_thrice"],overall_results["replan_results"]["events_seen_fourplus"],overall_results["replan_results"]["event_reward"],overall_results["replan_results"]["planner_reward"],overall_results["replan_results"]["percent_coverage"],overall_results["replan_results"]["event_max_revisit_time"],overall_results["replan_results"]["event_avg_revisit_time"],overall_results["replan_results"]["all_percent_coverage"],overall_results["replan_results"]["all_max_revisit_time"],overall_results["replan_results"]["all_avg_revisit_time"],
        #         overall_results["oracle_results"]["event_obs_count"],overall_results["oracle_results"]["events_seen_at_least_once"],overall_results["oracle_results"]["events_seen_once"],overall_results["oracle_results"]["events_seen_twice"],overall_results["oracle_results"]["events_seen_thrice"],overall_results["oracle_results"]["events_seen_fourplus"],overall_results["oracle_results"]["event_reward"],overall_results["oracle_results"]["planner_reward"],overall_results["oracle_results"]["percent_coverage"],overall_results["oracle_results"]["event_max_revisit_time"],overall_results["oracle_results"]["event_avg_revisit_time"],overall_results["oracle_results"]["all_percent_coverage"],overall_results["oracle_results"]["all_max_revisit_time"],overall_results["oracle_results"]["all_avg_revisit_time"],
        #         elapsed_time
        #     ]
        #     csvwriter.writerow(row)
        #     csvfile.close()
        metrics = {}
        metrics["reward"] = reward
        print(metrics)
        return metrics
    else:
        return "hello world"

if __name__ == '__main__':
	app.run(port='5000')