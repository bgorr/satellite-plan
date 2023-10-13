import numpy as np
import datetime
import csv
import os
import multiprocessing
from functools import partial
from tqdm import tqdm

def check_maneuver_feasibility(curr_angle,obs_angle,curr_time,obs_end_time):
    """
    Checks to see if the specified angle change violates the maximum slew rate constraint.
    """
    moved = False
    ss = 10.0
    # TODO add back FOV free visibility
    if(obs_end_time==curr_time):
        return False, False
    slew_rate = abs(obs_angle-curr_angle)/abs(obs_end_time-curr_time)/ss
    max_slew_rate = 1.0 # deg / s
    #slewTorque = 4 * abs(np.deg2rad(new_angle)-np.deg2rad(curr_angle))*0.05 / pow(abs(new_time-curr_time),2)
    #maxTorque = 4e-3
    transition_end_time = abs(obs_angle-curr_angle)/(max_slew_rate*ss) + curr_time
    moved = True
    return slew_rate < max_slew_rate, transition_end_time

def validate_plan(): 
    valid_plan = True 
    directory = "./missions/test_mission_5_reduced/orbit_data/"
    subdir = "sat0/"

    f = "datametrics_instru0_mode0_grid0.csv"
    visibilities = []
    with open(directory+subdir+f,newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        
        i = 0
        for row in spamreader:
            if i < 5:
                i=i+1
                continue
            row[2] = "0.0"
            row = [float(i) for i in row]
            row.append(subdir)
            visibilities.append(row)

    vis_windows = []
    i = 0
    while i < len(visibilities):
        continuous_visibilities = []
        visibility = visibilities[i]
        continuous_visibilities.append(visibility)
        start = visibility[0]
        end = visibility[0]
        while(i < len(visibilities)-1 and visibilities[i+1][0] == start):
            i += 1
        vis_done = False
        if i == len(visibilities)-1:
            break
        while not vis_done:
            vis_done = True
            num_steps = len(continuous_visibilities)
            while visibilities[i+1][0] == start+num_steps:
                if visibilities[i+1][1] == visibility[1]:
                    continuous_visibilities.append(visibilities[i+1])
                    end = visibilities[i+1][0]
                    vis_done = False
                if i == len(visibilities)-2:
                    break
                else:
                    i += 1
            num_steps = len(continuous_visibilities)
            if i == len(visibilities)-1:
                break
        vis_window = [start,end,visibility[3],visibility[4],visibility[-1]]
        vis_windows.append(vis_window)
        for cont_vis in continuous_visibilities:
            visibilities.remove(cont_vis)
        i = 0

    f = "plan_heuristic.csv"
    with open(directory+subdir+f,newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        observations = []
        i = 0
        for row in spamreader:
            if i < 1:
                i=i+1
                continue
            row = [float(i) for i in row]
            row.append(subdir)
            observations.append(row)
    for i in range(len(observations)-1):
        # obs file format: start, soonest, end, lat, lon, angle, reward
        # obs are in order
        obs = observations[i]
        next_obs = observations[i+1]
        if obs[0] >= next_obs[0]:
            print("Plan out of order or two obs start at the same time.")
            valid_plan = False
        feasible, next_time = check_maneuver_feasibility(obs[5],next_obs[5],obs[1],next_obs[2])
        if not feasible:
            print("Illegal transition.")
            valid_plan = False
    
    return valid_plan

def main():
    valid_plan = validate_plan()
    print(valid_plan)

if __name__ == "__main__":
    main()