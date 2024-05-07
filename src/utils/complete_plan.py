import os
import csv
import numpy as np
import datetime as datetime
import sys
sys.path.append(".")
from src.plan_mission import load_events, load_obs, load_rewards, load_satellites, chop_obs_list, close_enough


def save_plan(satellite,settings,flag):
    directory = settings["directory"] + "orbit_data/"
    with open(directory+satellite["orbitpy_id"]+'/replan_interval'+settings["planner"]+flag+'.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        plan = satellite["plan"]
        rows = []
        for obs in plan:
            row = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"],obs["pointing_angle"],obs["angle"],obs["reward"]]
            rows.append(row)

        for row in rows:
            csvwriter.writerow(row)

def complete_plan(flag,settings):
    directory = settings["directory"]+"orbit_data/"

    satellites = []

    for subdir in os.listdir(directory):
        satellite = {}
        if "comm" in subdir:
            continue
        if ".json" in subdir:
            continue
        if ".csv" in subdir:
            continue
        for f in os.listdir(directory+subdir):
            if "state_cartesian" in f:
                with open(directory+subdir+"/"+f,newline='') as csv_file:
                    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    states = []
                    i = 0
                    for row in spamreader:
                        if i < 5:
                            i=i+1
                            continue
                        row = [float(i) for i in row]
                        states.append(row)
                satellite["orbitpy_id"] = subdir
                satellite["states"] = states
                
            if "datametrics" in f:
                with open(directory+subdir+"/"+f,newline='') as csv_file:
                    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    visibilities = []
                    i = 0
                    for row in spamreader:
                        if i < 5:
                            i=i+1
                            continue
                        row[2] = "0.0"
                        row = [float(i) for i in row]
                        row.append(subdir)
                        visibilities.append(row)
                satellite["visibilities"] = visibilities
                #all_visibilities.extend(visibilities)

            if flag in f and settings["planner"] in f:
                with open(directory+subdir+"/"+f,newline='') as csv_file:
                    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    observations = []
                    for row in spamreader:
                        row = [float(i) for i in row]
                        row.append(subdir)
                        observations.append(row)
                    unique_observations = []
                    obs_end_times = []
                    for obs in observations:
                        if obs[1:4] not in obs_end_times:
                            obs_end_times.append(obs[1:4])
                            unique_observations.append(obs)
                    observations = unique_observations
                satellite["observations"] = observations

            if flag in f and settings["planner"] in f and "het" not in f and "oracle" not in f:
                with open(directory+subdir+"/"+f,newline='') as csv_file:
                    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    observations = []
                    for row in spamreader:
                        row = [float(i) for i in row]
                        row.append(subdir)
                        observations.append(row)
                    unique_observations = []
                    obs_end_times = []
                    for obs in observations:
                        if obs[1:4] not in obs_end_times:
                            obs_end_times.append(obs[1:4])
                            unique_observations.append(obs)
                    observations = unique_observations
                satellite["observations"] = observations            

        if "orbitpy_id" in satellite:
            satellites.append(satellite)

    for satellite in satellites:
        grid_locations = []
        with open(settings["point_grid"],'r') as csvfile:
            csvreader = csv.reader(csvfile,delimiter=',')
            next(csvfile)
            for row in csvreader:
                grid_locations.append([float(row[0]),float(row[1])])

    for satellite in satellites:
        obs_list = satellite["observations"]
        curr_ind = 0
        curr_angle = 0.0
        angle_list = []
        for i in range(int(settings["time"]["duration"]*86400/settings["time"]["step_size"])):
            if obs_list[curr_ind][0] > i:
                next_angle = obs_list[curr_ind][4]
                time_diff = obs_list[curr_ind][0] - i
                if time_diff > 1:
                    angle_diff = next_angle - curr_angle
                    angle_step = angle_diff/time_diff
                    curr_angle += angle_step
                    angle_list.append(curr_angle)
                elif time_diff <= 1:
                    curr_angle = next_angle
                    angle_list.append(curr_angle)
            elif obs_list[curr_ind][0] <= i:
                curr_angle = obs_list[curr_ind][4]
                angle_list.append(curr_angle)
                while obs_list[curr_ind][0] <= i and curr_ind < len(obs_list[:])-1:
                    curr_ind += 1
        satellite["obs_list"] = load_obs(satellite)
        observed_points = []
        satellite["curr_angle"] = 0
        for i in range(int(settings["time"]["duration"]*86400/settings["time"]["step_size"])):
            obs_list = chop_obs_list(satellite["obs_list"],i,i+1)
            pointing_option = angle_list[i]
            for obs in obs_list:
                if (np.abs(pointing_option-obs["angle"]) < settings["instrument"]["ffov"]/2):
                    observed_points.append(obs)
                    obs["pointing_angle"] = pointing_option
            satellite["curr_angle"] = pointing_option
        satellite["plan"] = observed_points
        save_plan(satellite, settings, flag)

def main():
    name = "madqn_test_fov_step_fullstate_expectedval"
    settings = {
        "name": name,
        "instrument": {
            "ffor": 60,
            "ffov": 5
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
            "num_sats_per_plane": 1,
            "num_planes": 1,
            "phasing_parameter": 1
        },
        "events": {
            "event_duration": 3600*6,
            "num_events": 10000,
            "event_clustering": "clustered"
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
        "plotting":{
            "plot_clouds": False,
            "plot_rain": False,
            "plot_duration": 0.1,
            "plot_interval": 10,
            "plot_obs": True
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
    complete_plan("init",settings)
    complete_plan("hom",settings)

if __name__ == "__main__":
    main()