import numpy as np
import os
import csv
import datetime

def greedy_lemaitre_planner(obs_list):
    """
    Based on the "greedy planner" from Lemaitre et al. Incorporates reward information and future options to decide observation plan.
    """
    estimated_reward = 100000
    rule_based_plan = []
    i = 0
    while i < 5:
        rule_based_plan = []
        more_actions = True
        last_obs = None
        curr_time = 0.0
        curr_angle = 0.0
        total_reward = 0.0
        obs_list_copy = obs_list.copy()
        while more_actions:
            best_obs = None
            maximum = 0.0
            actions = get_action_space(curr_time,curr_angle,obs_list,last_obs)
            if(len(actions) == 0):
                break
            for action in actions:
                duration = 8640
                rho = (duration - action["end"])/duration
                e = rho * estimated_reward
                adjusted_reward = np.abs(action["reward"]) + e
                if(adjusted_reward > maximum):
                    maximum = adjusted_reward
                    best_obs = action
            curr_time = best_obs["soonest"]
            curr_angle = best_obs["angle"]
            total_reward += best_obs["reward"]
            rule_based_plan.append(best_obs)
            last_obs = best_obs
        i += 1
        estimated_reward = total_reward
        obs_list = obs_list_copy
    return rule_based_plan

def greedy_fifo_planner(obs_list):
    return []

def get_action_space(curr_time,curr_angle,obs_list,last_obs):
    feasible_actions = []
    for obs in obs_list:
        if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
            continue
        if obs["start"] >= curr_time:
            feasible, transition_end_time = check_maneuver_feasibility(curr_angle,obs["angle"],curr_time,obs["end"]-1)
            if transition_end_time < obs["start"]:
                obs["soonest"] = obs["start"]+1
            else:
                obs["soonest"] = transition_end_time
            if feasible:
                feasible_actions.append(obs)
    return feasible_actions

def check_maneuver_feasibility(curr_angle,obs_angle,curr_time,obs_end_time):
    """
    Checks to see if the specified angle change violates the maximum slew rate constraint.
    """
    moved = False
    # TODO add back FOV free visibility
    if(obs_end_time==curr_time):
        return False, False
    slew_rate = abs(obs_angle-curr_angle)/abs(obs_end_time-curr_time)
    max_slew_rate = 1.0 # deg / s
    #slewTorque = 4 * abs(np.deg2rad(new_angle)-np.deg2rad(curr_angle))*0.05 / pow(abs(new_time-curr_time),2)
    #maxTorque = 4e-3
    transition_end_time = abs(obs_angle-curr_angle)/max_slew_rate + curr_time
    moved = True
    return slew_rate < max_slew_rate, transition_end_time

def plan_mission(settings):
    #directory = "./missions/test_mission/orbit_data/"
    directory = settings["directory"] + "orbit_data/"

    satellites = []

    for subdir in os.listdir(directory):
        if "comm" in subdir:
            continue
        if ".json" in subdir:
            continue
        satellite = {}
        for f in os.listdir(directory+subdir):
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
                        visibilities.append(row)
                satellite["visibilities"] = visibilities
                satellite["orbitpy_id"] = subdir

        satellites.append(satellite)

    for satellite in satellites:
        obs_list = []
        i = 0
        visibilities = satellite["visibilities"]
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

            time_window = {
                "location": {
                    "lat": visibility[3],
                    "lon": visibility[4]
                },
                "times": [x[0] for x in continuous_visibilities],
                "angles": [x[6] for x in continuous_visibilities],
                "start": start,
                "end": end,
                "angle": visibility[6],
                "reward": 1
            }
            if(time_window["location"]) is None:
                print(time_window)
            obs_list.append(time_window)
            for cont_vis in continuous_visibilities:
                visibilities.remove(cont_vis)
            i = 0
        plan = greedy_lemaitre_planner(obs_list)
        satellite["plan"] = plan
        with open(directory+satellite["orbitpy_id"]+'/plan.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in plan:
                row = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"]]
                csvwriter.writerow(row)

if __name__ == "__main__":
    settings = {
        "directory": "./missions/test_mission_2/",
        "step_size": 100,
        "duration": 0.2,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
    }
    plan_mission(settings)