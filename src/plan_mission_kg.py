import numpy as np
import os
import csv
import datetime
import multiprocessing
import random
from functools import partial
from tqdm import tqdm
from planners.mcts_planner import monte_carlo_tree_search
from planners.dp_planner import graph_search, graph_search_events, graph_search_events_interval, graph_search_kg, graph_search_ukge
from planners.milp_planner import milp_planner, milp_planner_interval
from planners.heuristic_planner import greedy_lemaitre_planner, greedy_lemaitre_planner_events, greedy_lemaitre_planner_events_interval
from planners.fifo_planner import fifo_planner, fifo_planner_events, fifo_planner_events_interval
from utils.planning_utils import close_enough
    
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

def decompose_plan(full_plan,satellites,settings):
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])
    for satellite in satellites:
        partial_plan = []
        for obs in full_plan:
            if obs[0] == satellite["orbitpy_id"]:
                partial_plan.append(obs[1:])
        with open(settings["directory"]+"orbit_data/"+satellite["orbitpy_id"]+'/plan_'+settings["planner"]+'.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in tqdm(partial_plan):
                obs_dict = {
                    "location": {
                        "lat": float(obs[0])*180/np.pi,
                        "lon": float(obs[1])*180/np.pi
                    },
                    "start": obs[2],
                    "end": obs[3],
                    "angle": float(obs[4])*180/np.pi,
                    "reward": obs[5]
                }
                obs = obs_dict
                for loc in grid_locations:
                    if within_fov(loc,obs["location"],np.min([settings["instrument"]["ffov"],settings["instrument"]["ffov"]]),500): # TODO fix hardcode
                        row = [obs["end"],obs["end"],loc[0],loc[1],obs["angle"],obs["reward"]]
                        csvwriter.writerow(row)
                if settings["instrument"]["ffov"] == 0:
                    row = [obs["end"],obs["end"],obs["location"]["lat"],obs["location"]["lon"],obs["angle"],obs["reward"]]
                    csvwriter.writerow(row)


def save_plan_w_fov(satellite,settings,grid_locations,flag):
    directory = settings["directory"] + "orbit_data/"
    with open(directory+satellite["orbitpy_id"]+'/replan_interval'+settings["planner"]+flag+'.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        plan = satellite["plan"]
        rows = []
        grid_locations = []
        with open(settings["point_grid"],'r') as csvfile:
            csvreader = csv.reader(csvfile,delimiter=',')
            next(csvfile)
            for row in csvreader:
                grid_locations.append([float(row[0]),float(row[1])])
        for obs in tqdm(plan):
            for loc in grid_locations:
                if within_fov(loc,obs["location"],np.min([settings["instrument"]["ffov"],settings["instrument"]["ffov"]]),500): # TODO fix hardcode
                    row = [obs["soonest"],obs["soonest"],loc[0],loc[1],obs["angle"],obs["reward"]]
                    rows.append(row)
            if settings["instrument"]["ffov"] == 0:
                row = [obs["soonest"],obs["soonest"],obs["location"]["lat"],obs["location"]["lon"],obs["angle"],obs["reward"]]
                rows.append(row)

        plan = unique(rows)
        for row in plan:
            csvwriter.writerow(row)

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
    
def chop_obs_list(obs_list,start,end):
    chopped_list = []
    for obs in obs_list:
        if obs["start"] > start and obs["end"] < end:
            chopped_list.append(obs)
    return chopped_list

def update_reward_dict(reward_dict,events,time,reward):
    for location in reward_dict.keys():
        events_per_location = []
        for event in events:
            if close_enough(location[0],location[1],event["location"]["lat"],event["location"]["lon"]):
                events_per_location.append(event)
        event_occurring = False
        for event in events_per_location:
            if (event["start"] <= time <= event["end"]):
                event_occurring = True
        if event_occurring:
            reward_dict[location]["reward"] = reward
        else:
            reward_dict[location]["reward"] = 0
        reward_dict[location]["last_updated"] = time
    return reward_dict

def update_reward_dict_het(reward_dict,events,time,reward,num_meas_types):
    for location in reward_dict.keys():
        events_per_location = []
        for event in events:
            if close_enough(location[0],location[1],event["location"]["lat"],event["location"]["lon"]):
                events_per_location.append(event)
        event_occurring = False
        for event in events_per_location:
            if (event["start"] <= time <= event["end"]):
                event_occurring = True
        if event_occurring:
            reward_dict[location]["rewards"] = [reward] * num_meas_types
        else:
            reward_dict[location]["rewards"] = [0] * num_meas_types
        reward_dict[location]["last_updated"] = time
    return reward_dict

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

        variables = ['Land surface temperature','Ocean Salinity','Ozone','Sea surface temperature','Sea-ice cover','Cloud cover','Cloud base height','Geoid','Land surface imagery']
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
            "reward": 1,
            "parameter": random.choice(variables)
        }
        if(time_window["location"]) is None:
            print(time_window)
        obs_list.append(time_window)
        for cont_vis in continuous_visibilities:
            visibilities.remove(cont_vis)
        i = 0
    if settings["planner"] == "heuristic":
        plan = greedy_lemaitre_planner(obs_list,settings)
    elif settings["planner"] == "fifo":
        plan = fifo_planner(obs_list,settings)
    elif settings["planner"] == "dp":
        if settings["kg_setting"] == "no_kg":
            plan = graph_search(obs_list,settings)
        elif settings["kg_setting"] == "det_kg":
            plan = graph_search_kg(obs_list,satellite["orbitpy_id"],settings)
        elif settings["kg_setting"] == "prob_kg":
            plan = graph_search_ukge(obs_list,satellite["orbitpy_id"],settings)
    elif settings["planner"] == "mcts":
        mcts = monte_carlo_tree_search()
        plan = mcts.do_search(obs_list,settings)
    elif settings["planner"] == "all":
        heuristic_plan = greedy_lemaitre_planner(obs_list,settings)
        fifo_plan = fifo_planner(obs_list,settings)
        dp_plan = graph_search(obs_list,settings)
        mcts = monte_carlo_tree_search()
        mcts_plan = mcts.do_search(obs_list,settings)
        satellite["plan"] = heuristic_plan
        grid_locations = []
        with open(settings["point_grid"],'r') as csvfile:
            csvreader = csv.reader(csvfile,delimiter=',')
            next(csvfile)
            for row in csvreader:
                grid_locations.append([float(row[0]),float(row[1])])
        with open(settings["directory"]+"orbit_data/"+satellite["orbitpy_id"]+'/plan_heuristic.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in tqdm(heuristic_plan):
                for loc in grid_locations:
                    if within_fov(loc,obs["location"],np.min([settings["instrument"]["ffov"],settings["instrument"]["ffov"]]),500): # TODO fix hardcode
                        row = [obs["soonest"],obs["soonest"],loc[0],loc[1],obs["angle"],obs["reward"]]
                        csvwriter.writerow(row)
        with open(settings["directory"]+"orbit_data/"+satellite["orbitpy_id"]+'/plan_dp.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in tqdm(dp_plan):
                for loc in grid_locations:
                    if within_fov(loc,obs["location"],np.min([settings["instrument"]["ffov"],settings["instrument"]["ffov"]]),500): # TODO fix hardcode
                        row = [obs["soonest"],obs["soonest"],loc[0],loc[1],obs["angle"],obs["reward"]]
                        csvwriter.writerow(row)
        with open(settings["directory"]+"orbit_data/"+satellite["orbitpy_id"]+'/plan_fifo.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in tqdm(fifo_plan):
                for loc in grid_locations:
                    if within_fov(loc,obs["location"],np.min([settings["instrument"]["ffov"],settings["instrument"]["ffov"]]),500): # TODO fix hardcode
                        row = [obs["soonest"],obs["soonest"],loc[0],loc[1],obs["angle"],obs["reward"]]
                        csvwriter.writerow(row)
        with open(settings["directory"]+"orbit_data/"+satellite["orbitpy_id"]+'/plan_mcts.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in tqdm(mcts_plan):
                for loc in grid_locations:
                    if within_fov(loc,obs["location"],np.min([settings["instrument"]["ffov"],settings["instrument"]["ffov"]]),500): # TODO fix hardcode
                        row = [obs["soonest"],obs["soonest"],loc[0],loc[1],obs["angle"],obs["reward"]]
                        csvwriter.writerow(row)
        return
    else:
        print("unsupported planner")
    satellite["plan"] = plan
    if not "point_grid" in settings:
        settings["point_grid"] = settings["directory"]+"orbit_data/grid0.csv"
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])
    with open(settings["directory"]+"orbit_data/"+satellite["orbitpy_id"]+'/plan_'+settings["planner"]+'.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for obs in tqdm(plan):
            for loc in grid_locations:
                if within_fov(loc,obs["location"],np.min([settings["instrument"]["ffov"],settings["instrument"]["ffov"]]),500): # TODO fix hardcode
                    row = [obs["soonest"],obs["soonest"],loc[0],loc[1],obs["angle"],obs["reward"],obs["parameter"]]
                    csvwriter.writerow(row)
            if settings["instrument"]["ffov"] == 0:
                row = [obs["soonest"],obs["soonest"],obs["location"]["lat"],obs["location"]["lon"],obs["angle"],obs["reward"],obs["parameter"]]
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
        if ".csv" in subdir:
            continue
        satellite = {}
        # already_planned = False
        # for f in os.listdir(directory+subdir):
        #     if "replan" in f and settings["planner"] in f:
        #         already_planned = True
        # if already_planned:
        #     continue
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
    if settings["planner"] == "milp":
        full_plan = []
        num_pieces = 100
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
                satellite["full_obs_list"] = obs_list
        for i in range(num_pieces):
            plan_start = i*86400*float(settings["time"]["duration"])/num_pieces/settings["time"]["step_size"]
            plan_end = (i+1)*86400*float(settings["time"]["duration"])/num_pieces/settings["time"]["step_size"]
            for satellite in satellites:
                obs_list = satellite["full_obs_list"].copy()
                satellite["obs_list"] = chop_obs_list(obs_list,plan_start,plan_end)
            partial_plan = milp_planner(satellites,settings)
            full_plan.extend(partial_plan)
        decompose_plan(full_plan,satellites,settings)
    else:
        # pool = multiprocessing.Pool()
        # pool.map(partial(plan_satellite, settings=settings), satellites)
        for satellite in satellites:
            plan_satellite(satellite,settings)
        
    print("Planned mission!")

def plan_mission_replan(settings):
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
        already_planned = False
        for f in os.listdir(directory+subdir):
            if "replan" in f:
                already_planned = True
        if already_planned:
            continue
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
                "start": float(row[2])/settings["time"]["step_size"],
                "end": (float(row[2])+float(row[3]))/settings["time"]["step_size"],
                "severity": float(row[4])
            }
            events.append(event)
    rewards = []
    reward_filename = './events/lakes/lake_event_points_reduced.csv'
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
    while elapsed_plan_time < float(settings["time"]["duration"])*86400:
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
            plan_end = float(settings["time"]["duration"])*86400
            planner_inputs = {
                "obs_list": new_obs_list.copy(),
                "plan_start": plan_start,
                "plan_end": plan_end,
                "events": events,
                "settings": settings
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
        if settings["planner"] == "heuristic":
            planner_output_list = pool.map(greedy_lemaitre_planner_events, planner_input_list)
        elif settings["planner"] == "fifo":
            planner_output_list = pool.map(fifo_planner_events, planner_input_list)
        elif settings["planner"] == "dp":
            planner_output_list = pool.map(graph_search_events, planner_input_list)
        elif settings["planner"] == "mcts":
            mcts = monte_carlo_tree_search()
            planner_output_list = pool.map(mcts.do_search_events, planner_input_list)
        
        for po in planner_output_list:
            end_times.append(po["end_time"])
            if po["updated_reward"] is not None:
                updated_reward_list.append(po["updated_reward"])
        soonest_end_time = float(settings["time"]["duration"])*86400 # TODO FIX
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
        if settings["planner"] == "heuristic":
            planner_output_list = pool.map(greedy_lemaitre_planner_events, planner_input_list)
        elif settings["planner"] == "fifo":
            planner_output_list = pool.map(fifo_planner_events, planner_input_list)
        elif settings["planner"] == "dp":
            planner_output_list = pool.map(graph_search_events, planner_input_list)
        elif settings["planner"] == "mcts":
            mcts = monte_carlo_tree_search()
            planner_output_list = pool.map(mcts.do_search_events, planner_input_list)
        for i in range(len(satellites)):
            if "plan" in satellites[i]:
                satellites[i]["plan"].extend(planner_output_list[i]["plan"])
            else:
                satellites[i]["plan"] = planner_output_list[i]["plan"]
        elapsed_plan_time = soonest_end_time
        print("Elapsed planning time: "+str(elapsed_plan_time))
    for satellite in satellites:
        with open(directory+satellite["orbitpy_id"]+'/replan_'+settings["planner"]+'.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            plan = satellite["plan"]
            rows = []
            for obs in plan:
                row = [obs["soonest"],obs["soonest"],obs["location"]["lat"],obs["location"]["lon"],obs["angle"],obs["reward"]]
                rows.append(row)
            plan = unique(rows)
            for row in plan:
                csvwriter.writerow(row)
    print("Planned mission with replans!")

def plan_mission_replan_interval(settings):
    print("Planning mission with replanning")
    directory = settings["directory"] + "orbit_data/"

    satellites = []

    for subdir in os.listdir(directory):
        if "comm" in subdir:
            continue
        if ".json" in subdir:
            continue
        if ".csv" in subdir:
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
    for event_filename in settings["event_csvs"]:
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
                    "start": float(row[2])/settings["time"]["step_size"],
                    "end": (float(row[2])+float(row[3]))/settings["time"]["step_size"],
                    "severity": float(row[4]),
                    "parameter": row[5]
                }
                events.append(event)
    rewards = []
    reward_filename = settings["point_grid"]
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
    reward_dict = {}
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])
    for loc in grid_locations:
        reward_dict[(np.round(loc[0],3),np.round(loc[1],3))] = {
            "last_updated": 0,
            "reward": 1
        }
    while elapsed_plan_time < float(settings["time"]["duration"])*86400/settings["time"]["step_size"]:
        updated_reward_list = []
        plan_interval = settings["planning_horizon"]/settings["time"]["step_size"]
        sharing_interval = settings["sharing_horizon"]/settings["time"]["step_size"]
        planner_input_list = []
        if settings["conops"] == "perfect_info":
            reward_dict = update_reward_dict(reward_dict,events,elapsed_plan_time,settings["rewards"]["reward"])
        for satellite in satellites:
            plan_start = elapsed_plan_time
            plan_end = plan_start+plan_interval
            sharing_end = plan_start+sharing_interval
            # Obs list update logic
            obs_list = satellite["obs_list"].copy()
            obs_list = chop_obs_list(obs_list,plan_start,plan_end)
            new_obs_list = []
            for obs in obs_list:
                if obs["start"] < elapsed_plan_time:
                    continue
                for location in reward_dict.keys():
                    if (obs["location"]["lat"],obs["location"]["lon"]) == location:
                        obs["reward"] = reward_dict[location]["reward"]
                        obs["last_updated"] = reward_dict[location]["last_updated"]
                new_obs_list.append(obs)
            new_obs_list = chop_obs_list(new_obs_list,plan_start,plan_end)
            planner_inputs = {
                "obs_list": new_obs_list.copy(),
                "plan_start": plan_start,
                "plan_end": plan_end,
                "sharing_end": sharing_end,
                "events": events,
                "settings": settings,
                "orbitpy_id": satellite["orbitpy_id"],
            }
            planner_input_list.append(planner_inputs)
        pool = multiprocessing.Pool()
        if settings["planner"] == "heuristic":
            planner_output_list = pool.map(greedy_lemaitre_planner_events_interval, planner_input_list)
        elif settings["planner"] == "fifo":
            planner_output_list = pool.map(fifo_planner_events_interval, planner_input_list)
        elif settings["planner"] == "dp":
            planner_output_list = pool.map(graph_search_events_interval, planner_input_list)
        elif settings["planner"] == "mcts":
            mcts = monte_carlo_tree_search()
            planner_output_list = pool.map(mcts.do_search_events_interval, planner_input_list)
        elif settings["planner"] == "milp":
            planner_output_list = milp_planner_interval(planner_input_list)
        for po in planner_output_list:
            if po["updated_rewards"] is not None:
                updated_reward_list.extend(po["updated_rewards"])
        if settings["conops"] == "onboard_processing":
            for updated_reward in updated_reward_list:
                key = (np.round(updated_reward["location"]["lat"],3),np.round(updated_reward["location"]["lon"],3))
                if key in reward_dict:
                    if updated_reward["last_updated"] > reward_dict[key]["last_updated"]:
                        reward_dict[key]["last_updated"] = updated_reward["last_updated"]
                        reward_dict[key]["reward"] = updated_reward["reward"]
                else:
                    reward_dict[key] = {
                        "last_updated": updated_reward["last_updated"],
                        "reward": updated_reward["reward"]
                    }
            rewards = []
            for location in reward_dict.keys():
                rewards.append((location[0],location[1],reward_dict[location]["reward"]))
                reward_dict[location]["reward"] += settings["rewards"]["reward_increment"]
                if (reward_dict[location]["last_updated"] + settings["events"]["event_duration"]/settings["time"]["step_size"]) < elapsed_plan_time:
                    reward_dict[location]["last_updated"] = elapsed_plan_time
                    reward_dict[location]["reward"] = 1
            if not os.path.exists(settings["directory"]+'reward_grids/'):
                os.mkdir(settings["directory"]+'reward_grids/')
            with open(settings["directory"]+'reward_grids/step_'+str(elapsed_plan_time)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for reward in rewards:
                    csvwriter.writerow(reward)
        for i in range(len(satellites)):
            full_plan = planner_output_list[i]["plan"]
            trimmed_plan = chop_obs_list(full_plan,plan_start,sharing_end)
            if "plan" in satellites[i]:
                satellites[i]["plan"].extend(trimmed_plan)
            else:
                satellites[i]["plan"] = trimmed_plan
        elapsed_plan_time += sharing_interval
        print("Elapsed planning time: "+str(elapsed_plan_time))
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])
    pool.map(partial(save_plan_w_fov, settings=settings, grid_locations=grid_locations, flag="hom"), satellites)
    print("Planned mission with replans at interval!")

def plan_mission_replan_interval_het(settings):
    print("Planning mission with replanning")
    directory = settings["directory"] + "orbit_data/"

    satellites = []

    for subdir in os.listdir(directory):
        if "comm" in subdir:
            continue
        if ".json" in subdir:
            continue
        satellite = {}
        satellite_name_dict = {}
        for i in range(settings["constellation"]["num_sats_per_plane"]*settings["constellation"]["num_planes"]):
            if i < settings["constellation"]["num_sats_per_plane"]*settings["constellation"]["num_planes"]/settings["num_meas_types"]:
                meas_type = 0
            elif i < 2*settings["constellation"]["num_sats_per_plane"]*settings["constellation"]["num_planes"]/settings["num_meas_types"]:
                meas_type = 1
            elif i < 3*settings["constellation"]["num_sats_per_plane"]*settings["constellation"]["num_planes"]/settings["num_meas_types"]:
                meas_type = 2
            elif i < 4*settings["constellation"]["num_sats_per_plane"]*settings["constellation"]["num_planes"]/settings["num_meas_types"]:
                meas_type = 3
            satellite_name_dict["sat"+str(i)] = meas_type
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
    for event_filename in settings["event_csvs"]:
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
                    "start": float(row[2])/settings["time"]["step_size"],
                    "end": (float(row[2])+float(row[3]))/settings["time"]["step_size"],
                    "severity": float(row[4])
                }
                events.append(event)
    rewards = []
    reward_filename = settings["point_grid"]
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
    reward_dict = {}
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])
    for loc in grid_locations:
        reward_dict[(np.round(loc[0],3),np.round(loc[1],3))] = {
            "last_updated": 0,
            "rewards": [1] * settings["num_meas_types"],
            "obs_count": [0] * settings["num_meas_types"]
        }
    while elapsed_plan_time < float(settings["time"]["duration"])*86400/settings["time"]["step_size"]:
        updated_reward_list = []
        plan_interval = settings["planning_horizon"]/settings["time"]["step_size"]
        sharing_interval = settings["sharing_horizon"]/settings["time"]["step_size"]
        planner_input_list = []
        if settings["conops"] == "perfect_info":
            reward_dict = update_reward_dict_het(reward_dict,events,elapsed_plan_time,settings["rewards"]["reward"],settings["num_meas_types"])
        for satellite in satellites:
            plan_start = elapsed_plan_time
            plan_end = plan_start+plan_interval
            sharing_end = plan_start+sharing_interval
            # Obs list update logic
            obs_list = satellite["obs_list"].copy()
            obs_list = chop_obs_list(obs_list,plan_start,plan_end)
            new_obs_list = []
            for obs in obs_list:
                if obs["start"] < elapsed_plan_time:
                    continue
                for location in reward_dict.keys():
                    if (obs["location"]["lat"],obs["location"]["lon"]) == location:
                        obs["reward"] = reward_dict[location]["rewards"][satellite_name_dict[satellite["orbitpy_id"]]] 
                        obs["last_updated"] = reward_dict[location]["last_updated"]
                new_obs_list.append(obs)
            new_obs_list = chop_obs_list(new_obs_list,plan_start,plan_end)
            planner_inputs = {
                "obs_list": new_obs_list.copy(),
                "plan_start": plan_start,
                "plan_end": plan_end,
                "sharing_end": sharing_end,
                "events": events,
                "settings": settings,
                "orbitpy_id": satellite["orbitpy_id"]
            }
            planner_input_list.append(planner_inputs)
        pool = multiprocessing.Pool()
        if settings["planner"] == "heuristic":
            planner_output_list = pool.map(greedy_lemaitre_planner_events_interval, planner_input_list)
        elif settings["planner"] == "fifo":
            planner_output_list = pool.map(fifo_planner_events_interval, planner_input_list)
        elif settings["planner"] == "dp":
            planner_output_list = pool.map(graph_search_events_interval, planner_input_list)
        elif settings["planner"] == "mcts":
            mcts = monte_carlo_tree_search()
            planner_output_list = pool.map(mcts.do_search_events_interval, planner_input_list)
        elif settings["planner"] == "milp":
            planner_output_list = milp_planner_interval(planner_input_list)
        for po in planner_output_list:
            if po["updated_rewards"] is not None:
                updated_reward_list.extend(po["updated_rewards"])
        if settings["conops"] == "onboard_processing":
            for updated_reward in updated_reward_list:
                key = (np.round(updated_reward["location"]["lat"],3),np.round(updated_reward["location"]["lon"],3))
                if key in reward_dict:
                    if updated_reward["last_updated"] > reward_dict[key]["last_updated"]:
                        reward_dict[key]["last_updated"] = updated_reward["last_updated"]
                        for i in range(settings["num_meas_types"]):
                            if updated_reward["reward"] == 0:
                                reward_dict[key]["rewards"][i] = updated_reward["reward"]
                            elif satellite_name_dict[updated_reward["orbitpy_id"]] == i:
                                reward_dict[key]["rewards"][i] = 0
                                #reward_dict[key]["obs_count"][i] += 1
                            elif reward_dict[key]["obs_count"][i] == 0:
                                reward_dict[key]["rewards"][i] = updated_reward["reward"]
                            else:
                                print("???")
                else:
                    print("Error: trying to add a new location to the reward grid.")

            rewards = []
            for location in reward_dict.keys():
                rewards.append((location[0],location[1],reward_dict[location]["rewards"]))
                reward_dict[location]["rewards"] = [x+settings["rewards"]["reward_increment"] for x in reward_dict[location]["rewards"]]
                if (reward_dict[location]["last_updated"] + settings["events"]["event_duration"]/settings["time"]["step_size"]) < elapsed_plan_time:
                    reward_dict[location]["last_updated"] = elapsed_plan_time
                    reward_dict[location]["rewards"] = [1] * settings["num_meas_types"]
                    reward_dict[location]["obs_count"] = [0] * settings["num_meas_types"]
                for i in range(settings["num_meas_types"]):
                    count = 0
                    if reward_dict[location]["obs_count"][i] > 0:
                        reward_dict[location]["rewards"][i] = 0
                        count += 1
                if count == settings["num_meas_types"]:
                    reward_dict[location]["last_updated"] = elapsed_plan_time
                    reward_dict[location]["rewards"] = [1] * settings["num_meas_types"]
                    reward_dict[location]["obs_count"] = [0] * settings["num_meas_types"]
            if not os.path.exists(settings["directory"]+'reward_grids_het/'):
                os.mkdir(settings["directory"]+'reward_grids_het/')
            with open(settings["directory"]+'reward_grids_het/step_'+str(elapsed_plan_time)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for reward in rewards:
                    csvwriter.writerow(reward)
        for i in range(len(satellites)):
            full_plan = planner_output_list[i]["plan"]
            trimmed_plan = chop_obs_list(full_plan,plan_start,sharing_end)
            if "plan" in satellites[i]:
                satellites[i]["plan"].extend(trimmed_plan)
            else:
                satellites[i]["plan"] = trimmed_plan
        elapsed_plan_time += sharing_interval
        print("Elapsed planning time: "+str(elapsed_plan_time))
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])
    pool.map(partial(save_plan_w_fov, settings=settings, grid_locations=grid_locations, flag="het"), satellites)
    print("Planned mission with replans at interval (het)!")

if __name__ == "__main__":
    mission_name = "milp_test"
    cross_track_ffor = 90 # deg
    along_track_ffor = 90 # deg
    cross_track_ffov = 1 # deg
    along_track_ffov = 1 # deg
    agility = 0.01 # deg/s
    num_planes = 4
    num_sats_per_plane = 4
    var = 4 # deg lat/lon
    num_points_per_cell = 10
    simulation_step_size = 10 # seconds
    simulation_duration = 1 # days
    event_frequency = 1e-5 # events per second
    event_duration = 21600 # second

    settings = {
        "directory": "./missions/milp_test/",
        "step_size": simulation_step_size,
        "duration": simulation_duration,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "point_grid": "./coverage_grids/"+mission_name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+mission_name+"/events.csv"],
        "cross_track_ffor": 30,
        "along_track_ffor": 30,
        "cross_track_ffov": 0,
        "along_track_ffov": 0,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False,
        "planner": "milp",
        "reward": 10,
        "reobserve_reward": 2,
    }
    plan_mission(settings)
    #plan_mission_replan_interval(settings)
    #plan_mission_mcts(settings)
    #plan_mission_dp(settings)
    #plan_mission_fifo(settings)
    #plan_mission_with_replanning_intervals(settings)