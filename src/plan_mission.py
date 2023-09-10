import numpy as np
import os
import csv
import datetime
import multiprocessing
from functools import partial
from tqdm import tqdm

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.01:
        return True
    else:
        return False
    
def unique(lakes):
    lakes = np.asarray(lakes)
    return np.unique(lakes,axis=0)

def repair_plan(plan):
    i = 0
    while i < len(plan)-1:
        if plan[i][1] > plan[i+1][0]:
            plan.remove(plan[i+1])
        else:
            i = i+1
    return plan

def within_fov(loc_array,loc_dict,angle,orbit_height_km):
    lat1 = np.deg2rad(loc_array[0])
    lon1 = np.deg2rad(loc_array[1])
    lat2 = np.deg2rad(loc_dict["lat"])
    lon2 = np.deg2rad(loc_dict["lon"])
    h = 0 # height above in m
    a = 6378e3 # m
    e = 0
    N_phi1 = a/np.sqrt(1-e**2*np.sin(lat1)**2)
    x1 = (N_phi1+h)*np.cos(lat1)*np.cos(lon1)
    y1 = (N_phi1+h)*np.cos(lat1)*np.sin(lon1)
    z1 = ((1-e**2)*N_phi1+h)*np.sin(lat1)
    N_phi2 = a/np.sqrt(1-e**2*np.sin(lat2)**2)
    x2 = (N_phi2+h)*np.cos(lat2)*np.cos(lon2)
    y2 = (N_phi2+h)*np.cos(lat2)*np.sin(lon2)
    z2 = ((1-e**2)*N_phi2+h)*np.sin(lat2)
    dist = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
    if np.arctan(dist/((orbit_height_km)*1e3)) < np.deg2rad(angle/2):
        return True
    else:
        return False

def greedy_lemaitre_planner(obs_list,settings):
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
            actions = get_action_space(curr_time,curr_angle,obs_list,last_obs,settings)
            if(len(actions) == 0):
                break
            for action in actions:
                duration = 86400/settings["step_size"]
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

def greedy_lemaitre_planner_events(planner_inputs):
    """
    Based on the "greedy planner" from Lemaitre et al. Incorporates reward information and future options to decide observation plan.
    """
    obs_list = planner_inputs["obs_list"]
    plan_start = planner_inputs["plan_start"]
    plan_end = planner_inputs["plan_end"]
    events = planner_inputs["events"]
    estimated_reward = 100000
    rule_based_plan = []
    i = 0
    while i < 5:
        rule_based_plan = []
        more_actions = True
        last_obs = None
        curr_time = plan_start
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
                duration = 86400
                rho = (duration - action["end"])/duration
                e = rho * estimated_reward
                adjusted_reward = np.abs(action["reward"]) + e
                if(adjusted_reward > maximum):
                    maximum = adjusted_reward
                    best_obs = action
            if best_obs is None:
                break
            curr_time = best_obs["soonest"]
            curr_angle = best_obs["angle"]
            total_reward += best_obs["reward"]
            rule_based_plan.append(best_obs)
            for event in events:
                if close_enough(best_obs["location"]["lat"],best_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                    # print("close enough!")
                    # print(event)
                    # print(best_obs)
                    if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                        updated_reward = { 
                            "reward": best_obs["reward"],
                            "location": best_obs["location"],
                            "last_updated": curr_time 
                        }
                        planner_outputs = {
                            "plan": rule_based_plan,
                            "end_time": curr_time,
                            "updated_reward": updated_reward
                        }
                        return planner_outputs
            last_obs = best_obs
            if curr_time > plan_end:
                break
        i += 1
        estimated_reward = total_reward
        obs_list = obs_list_copy
    planner_outputs = {
                        "plan": rule_based_plan,
                        "end_time": plan_end,
                        "updated_reward": None
                    }
    return planner_outputs

def greedy_lemaitre_planner_events_interval(planner_inputs):
    """
    Based on the "greedy planner" from Lemaitre et al. Incorporates reward information and future options to decide observation plan.
    """
    obs_list = planner_inputs["obs_list"]
    plan_start = planner_inputs["plan_start"]
    plan_end = planner_inputs["plan_end"]
    events = planner_inputs["events"]
    estimated_reward = 100000
    rule_based_plan = []
    i = 0
    while i < 5:
        rule_based_plan = []
        updated_rewards = []
        more_actions = True
        last_obs = None
        curr_time = plan_start
        curr_angle = 0.0
        total_reward = 0.0
        obs_list_copy = obs_list.copy()
        location_list = []
        while more_actions:
            best_obs = None
            maximum = 0.0
            actions = get_action_space(curr_time,curr_angle,obs_list,last_obs)
            if(len(actions) == 0):
                break
            for action in actions:
                if action["location"] in location_list:
                    continue
                duration = 86400 # TODO FIX
                rho = (duration - action["end"])/duration
                e = rho * estimated_reward
                adjusted_reward = np.abs(action["reward"]) + e
                if(adjusted_reward > maximum):
                    maximum = adjusted_reward
                    best_obs = action
            if best_obs is None:
                break
            location_list.append(best_obs["location"]) # already seen by this sat
            curr_time = best_obs["soonest"]
            curr_angle = best_obs["angle"]
            total_reward += best_obs["reward"]
            rule_based_plan.append(best_obs)
            for event in events:
                if close_enough(best_obs["location"]["lat"],best_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                    # print("close enough!")
                    # print(event)
                    # print(best_obs)
                    if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                        updated_reward = { 
                            "reward": event["severity"]*10,
                            "location": best_obs["location"],
                            "last_updated": curr_time 
                        }
                        updated_rewards.append(updated_reward)
            last_obs = best_obs
            if curr_time > plan_end:
                break
        i += 1
        estimated_reward = total_reward
        obs_list = obs_list_copy
    planner_outputs = {
                        "plan": rule_based_plan,
                        "end_time": plan_end,
                        "updated_rewards": updated_rewards
                    }
    return planner_outputs


def greedy_fifo_planner(obs_list):
    fifo_plan = []
    curr_time = 0.0
    curr_angle = 0.0
    last_obs = None
    while True:
        actions = get_action_space(curr_time,curr_angle,obs_list,last_obs)
        if len(actions) == 0:
            break
        next_obs = actions[0]
        fifo_plan.append(next_obs)
        curr_time = next_obs["soonest"]
        curr_angle = next_obs["angle"]
    return fifo_plan

def get_action_space(curr_time,curr_angle,obs_list,last_obs,settings):
    feasible_actions = []
    for obs in obs_list:
        if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
            continue
        if obs["start"] > curr_time:
            feasible, transition_end_time = check_maneuver_feasibility(curr_angle,obs["angle"],curr_time,obs["end"]-1,settings)
            if transition_end_time < obs["start"]:
                obs["soonest"] = obs["start"]
            else:
                obs["soonest"] = transition_end_time
            if feasible:
                feasible_actions.append(obs)
        if len(feasible_actions) > 10: # THIS IS NOT A GOOD IDEA BUT SHOULD HELP RUNTIME TODO
            break
    return feasible_actions

def check_maneuver_feasibility(curr_angle,obs_angle,curr_time,obs_end_time,settings):
    """
    Checks to see if the specified angle change violates the maximum slew rate constraint.
    """
    moved = False
    # TODO add back FOV free visibility
    if(obs_end_time==curr_time):
        return False, False
    slew_rate = abs(obs_angle-curr_angle)/abs(obs_end_time-curr_time)
    max_slew_rate = settings["agility"] # deg / s
    #slewTorque = 4 * abs(np.deg2rad(new_angle)-np.deg2rad(curr_angle))*0.05 / pow(abs(new_time-curr_time),2)
    #maxTorque = 4e-3
    transition_end_time = abs(obs_angle-curr_angle)/max_slew_rate + curr_time
    moved = True
    return slew_rate < max_slew_rate, transition_end_time

def plan_satellite(satellite,settings):
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
    plan = greedy_lemaitre_planner(obs_list,settings)
    print(len(plan))
    satellite["plan"] = plan
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])
    with open(settings["directory"]+"orbit_data/"+satellite["orbitpy_id"]+'/plan.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for obs in tqdm(plan):
            for loc in grid_locations:
                if within_fov(loc,obs["location"],np.min([settings["cross_track_ffov"],settings["along_track_ffov"]]),500): # TODO fix hardcode
                    row = [obs["start"],obs["end"],loc[0],loc[1]]
                    csvwriter.writerow(row)
            row = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"]]
            csvwriter.writerow(row)
def plan_mission(settings):
    print("Planning mission")
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
    pool = multiprocessing.Pool()
    pool.map(partial(plan_satellite, settings=settings), satellites)
    # for satellite in satellites:
    #     plan_satellite(satellite,settings)
        
    print("Planned mission!")

def plan_mission_with_replanning(settings):
    print("Planning mission with replanning")
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
    events = []
    event_filename = './events/lakes/all_events.csv'
    with open(event_filename,newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            event = {
                "location": {
                    "lat": float(row[0]),
                    "lon": float(row[1]),
                },
                "start": float(row[2]),
                "end": float(row[2])+float(row[3]),
                "severity": float(row[4])
            }
            events.append(event)
    rewards = []
    reward_filename = './events/lakes/lake_event_points.csv'
    with open(reward_filename,newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            reward = {
                "location": {
                    "lat": float(row[0]),
                    "lon": float(row[1]),
                },
                "reward": 1.0,
                "last_updated": 0.0
            }
            rewards.append(reward)
    for satellite in satellites:
            obs_list = []
            i = 0
            visibilities = satellite["visibilities"].copy()
            while i < len(visibilities):
                continuous_visibilities = []
                visibility = visibilities[i]
                continuous_visibilities.append(visibility)
                start = float(visibility[0])*settings["step_size"]
                end = float(visibility[0])*settings["step_size"]
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
                    "times": [float(x[0])*settings["step_size"] for x in continuous_visibilities],
                    "angles": [x[6] for x in continuous_visibilities],
                    "start": start,
                    "end": end,
                    "angle": visibility[6],
                    "reward": 1,
                    "last_updated": 0.0
                }
                if(time_window["location"]) is None:
                    print(time_window)
                obs_list.append(time_window)
                for cont_vis in continuous_visibilities:
                    visibilities.remove(cont_vis)
                i = 0
            satellite["obs_list"] = obs_list
    elapsed_plan_time = 0
    reward_update_locations = []
    while elapsed_plan_time < float(settings["duration"])*86400:
        end_times = []
        updated_reward_list = []
        planner_input_list = []
        for satellite in satellites:
            obs_list = satellite["obs_list"].copy()
            new_obs_list = []
            for obs in obs_list:
                for rul in reward_update_locations:
                    if obs["location"] == rul["location"]:
                        obs["reward"] = rul["reward"]
                if obs["start"] < elapsed_plan_time:
                    continue
                new_obs_list.append(obs)
            plan_start = elapsed_plan_time
            plan_end = float(settings["duration"])*86400
            planner_inputs = {
                "obs_list": new_obs_list.copy(),
                "plan_start": plan_start,
                "plan_end": plan_end,
                "events": events
            }
            satellite["obs_list"] = new_obs_list.copy()
            planner_input_list.append(planner_inputs)
            # planner_outputs = greedy_lemaitre_planner_events(planner_inputs)
            # updated_reward = planner_outputs["updated_reward"]
            # plan = planner_outputs["plan"]
            # end_time = planner_outputs["end_time"]
            # end_times.append(end_time)
            # if updated_reward is not None:
            #     updated_reward_list.append(updated_reward)
        pool = multiprocessing.Pool()
        planner_output_list = pool.map(greedy_lemaitre_planner_events, planner_input_list)
        for po in planner_output_list:
            end_times.append(po["end_time"])
            if po["updated_reward"] is not None:
                updated_reward_list.append(po["updated_reward"])
        soonest_end_time = float(settings["duration"])*86400 # TODO FIX
        for time in end_times:
            if time < soonest_end_time:
                soonest_end_time = time
        reward_update_locations = []
        for updated_reward in updated_reward_list:
            reward_update_locations.append({"location": updated_reward["location"],
                                            "reward": updated_reward["reward"]})
            for i in range(len(rewards)):
                if updated_reward["location"] == rewards[i]["location"]:
                    rewards[i] = updated_reward
        planner_input_list = []
        for satellite in satellites:
            planner_inputs = {
                "obs_list": satellite["obs_list"],
                "plan_start": plan_start,
                "plan_end": soonest_end_time,
                "events": events
            }
            planner_input_list.append(planner_inputs)
        planner_output_list = pool.map(greedy_lemaitre_planner_events, planner_input_list)
        for i in range(len(satellites)):
            if "plan" in satellites[i]:
                satellites[i]["plan"].extend(planner_output_list[i]["plan"])
            else:
                satellites[i]["plan"] = planner_output_list[i]["plan"]
        elapsed_plan_time = soonest_end_time
        print("Elapsed planning time: "+str(elapsed_plan_time))
    for satellite in satellites:
        with open(directory+satellite["orbitpy_id"]+'/plan_w_replan.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            plan = satellite["plan"]
            rows = []
            for obs in plan:
                row = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"]]
                rows.append(row)
            plan = unique(rows)
            for row in plan:
                csvwriter.writerow(row)
    print("Planned mission with replans!")

def plan_mission_with_replanning_intervals(settings):
    print("Planning mission with replanning")
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
    events = []
    event_filename = './events/lakes/all_events.csv'
    with open(event_filename,newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            event = {
                "location": {
                    "lat": float(row[0]),
                    "lon": float(row[1]),
                },
                "start": float(row[2]),
                "end": float(row[2])+float(row[3]),
                "severity": float(row[4])
            }
            events.append(event)
    rewards = []
    reward_filename = './events/lakes/lake_event_points.csv'
    with open(reward_filename,newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            reward = {
                "location": {
                    "lat": float(row[0]),
                    "lon": float(row[1]),
                },
                "reward": 1.0,
                "last_updated": 0.0
            }
            rewards.append(reward)
    for satellite in satellites:
            obs_list = []
            i = 0
            visibilities = satellite["visibilities"].copy()
            while i < len(visibilities):
                continuous_visibilities = []
                visibility = visibilities[i]
                continuous_visibilities.append(visibility)
                start = float(visibility[0])*settings["step_size"]
                end = float(visibility[0])*settings["step_size"]
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
                    "times": [float(x[0])*settings["step_size"] for x in continuous_visibilities],
                    "angles": [x[6] for x in continuous_visibilities],
                    "start": start,
                    "end": end,
                    "angle": visibility[6],
                    "reward": 1,
                    "last_updated": 0.0
                }
                if(time_window["location"]) is None:
                    print(time_window)
                obs_list.append(time_window)
                for cont_vis in continuous_visibilities:
                    visibilities.remove(cont_vis)
                i = 0
            satellite["obs_list"] = obs_list
    elapsed_plan_time = 0
    reward_update_locations = []
    while elapsed_plan_time < float(settings["duration"])*86400:
        updated_reward_list = []
        planner_input_list = []
        for satellite in satellites:
            obs_list = satellite["obs_list"].copy()
            new_obs_list = []
            for obs in obs_list:
                for rul in reward_update_locations:
                    if obs["location"] == rul["location"]:
                        obs["reward"] = rul["reward"]
                        obs["last_updated"] = elapsed_plan_time
                if (obs["last_updated"] + 2000) < elapsed_plan_time:
                    obs["last_updated"] = elapsed_plan_time
                    obs["reward"] = 1
                if obs["start"] < elapsed_plan_time:
                    continue
                new_obs_list.append(obs)
            plan_start = elapsed_plan_time
            interval = 500
            plan_end = plan_start+interval
            planner_inputs = {
                "obs_list": new_obs_list.copy(),
                "plan_start": plan_start,
                "plan_end": plan_end,
                "events": events
            }
            planner_input_list.append(planner_inputs)
        pool = multiprocessing.Pool()
        planner_output_list = pool.map(greedy_lemaitre_planner_events_interval, planner_input_list)
        for po in planner_output_list:
            if po["updated_rewards"] is not None:
                updated_reward_list.extend(po["updated_rewards"])
        reward_update_locations = []
        for updated_reward in updated_reward_list:
            reward_update_locations.append({"location": updated_reward["location"],
                                            "reward": updated_reward["reward"],
                                            "last_updated": plan_end})
            for i in range(len(rewards)):
                if updated_reward["location"] == rewards[i]["location"]:
                    rewards[i] = updated_reward
        for i in range(len(satellites)):
            if "plan" in satellites[i]:
                satellites[i]["plan"].extend(planner_output_list[i]["plan"])
            else:
                satellites[i]["plan"] = planner_output_list[i]["plan"]
        elapsed_plan_time += interval
        print("Elapsed planning time: "+str(elapsed_plan_time))
    for satellite in satellites:
        with open(directory+satellite["orbitpy_id"]+'/plan_w_replan_interval.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            plan = satellite["plan"]
            rows = []
            for obs in plan:
                row = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"]]
                rows.append(row)
            #plan = repair_plan(rows)
            plan = unique(rows)
            for row in plan:
                csvwriter.writerow(row)
    print("Planned mission with replans at interval!")

if __name__ == "__main__":
    mission_name = "experiment0"
    cross_track_ffor = 60 # deg
    along_track_ffor = 60 # deg
    cross_track_ffov = 10 # deg
    along_track_ffov = 10 # deg
    agility = 1 # deg/s
    num_planes = 5 # deg/s
    num_sats_per_plane = 10 # deg/s
    var = 1 # deg lat/lon
    num_points_per_cell = 10
    simulation_step_size = 10 # seconds
    simulation_duration = 1 # days
    event_frequency = 1e-5 # events per second
    event_duration = 3600 # seconds
    settings = {
        "directory": "./missions/"+mission_name+"/",
        "step_size": simulation_step_size,
        "duration": simulation_duration,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "point_grid": "./coverage_grids/"+mission_name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+mission_name+"/events.csv"],
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": True,
        "plot_duration": 2/24,
        "plot_interval": 10,
        "plot_location": "./missions/"+mission_name+"/plots/",
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False
    }
    plan_mission(settings)
    #plan_mission_with_replanning_intervals(settings)