import numpy as np
import datetime
import csv
import os
import multiprocessing
from functools import partial
from tqdm import tqdm

def unique(lakes):
    lakes = np.asarray(lakes)[:,0:1]
    return np.unique(lakes,axis=0)

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.001:
        return True
    else:
        return False

def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))

def compute_max_revisit_time(start,end,observations,settings):
    # only computing based on starts so that I don't have to do start/stop tracking
    # TODO stop being lazy
    start_list = []
    start_list.append(start)
    start_list.append(end)
    for obs in observations:
        start_list.append(obs[0]*settings["time"]["step_size"])
    start_list = np.asarray(start_list)
    start_list = np.sort(start_list)
    gaps = []
    for i in range(len(start_list)-1):
        gaps.append(start_list[i+1]-start_list[i])
    gaps = np.asarray(gaps)
    return np.max(gaps)

def compute_avg_revisit_time(start,end,observations,settings):
    # only computing based on starts so that I don't have to do start/stop tracking
    # TODO stop being lazy
    start_list = []
    start_list.append(start)
    start_list.append(end)  
    for obs in observations:
        start_list.append(obs[0]*settings["time"]["step_size"])
    start_list = np.asarray(start_list)
    start_list = np.sort(start_list)
    gaps = []
    for i in range(len(start_list)-1):
        gaps.append(start_list[i+1]-start_list[i])
    gaps = np.asarray(gaps)
    return np.average(gaps)

def compute_statistics_pieces(input):
    events = input["events"]
    observations = input["observations"]
    settings = input["settings"]
    event_obs_pairs = []
    num_event_obs = 0
    obs_per_event_list = []
    #event_duration = settings["events"]["event_duration"]
    ss = settings["time"]["step_size"]
    cumulative_event_reward = 0
    cumulative_plan_reward = 0
    for event in events:
        obs_per_event = 0
        last_obs_time = None
        event_reward = float(event[4])
        for obs in observations:
            if obs[0] > (float(event[2])/ss+float(event[3])/ss):
                break
            if close_enough(obs[2],obs[3],float(event[0]),float(event[1])):
                if ((float(event[2])/ss) < obs[0] < (float(event[2])/ss+float(event[3])/ss)) or ((float(event[2])/ss) < obs[1] < (float(event[2])/ss+float(event[3])/ss)):
                    if last_obs_time is None:
                        event_obs_pair = {
                            "event": event,
                            "obs": obs
                        }
                        event_obs_pairs.append(event_obs_pair)
                        cumulative_event_reward += event_reward
                        cumulative_plan_reward += settings["rewards"]["reward"]
                        obs_per_event += 1
                        num_event_obs += 1
                        last_obs_time = obs[0]
                    elif obs[0] - last_obs_time > 2:
                        event_obs_pair = {
                            "event": event,
                            "obs": obs
                        }
                        event_obs_pairs.append(event_obs_pair)
                        cumulative_event_reward += event_reward
                        cumulative_plan_reward += settings["rewards"]["reward"]
                        obs_per_event += 1
                        num_event_obs += 1
                        last_obs_time = obs[0]
        obs_per_event_list.append(obs_per_event)

    output = {}
    output["event_obs_pairs"] = event_obs_pairs
    output["num_event_obs"] = num_event_obs
    output["obs_per_event_list"] = obs_per_event_list
    output["cumulative_event_reward"] = cumulative_event_reward
    output["cumulative_plan_reward"] = cumulative_plan_reward
    return output

def compute_statistics(events,obs,grid_locations,settings):
    obs.sort(key=lambda obs: obs[0])
    event_chunks = list(chunks(events,25))
    pool = multiprocessing.Pool()
    input_list = []
    output_list = []
    for i in range(len(event_chunks)):
        input = dict()
        input["events"] = event_chunks[i]
        input["observations"] = obs
        input["settings"] = settings
        output_list.append(compute_statistics_pieces(input))
    #output_list = pool.map(compute_statistics_pieces, input_list)
    #print("right before map")
    #output_list = list(tqdm(pool.map(compute_statistics_pieces, input_list)))
    all_events_count = 0
    planner_reward = 0
    event_reward = 0
    event_obs_pairs = []
    obs_per_event_list = []

    for output in output_list:
        event_reward += output["cumulative_event_reward"]
        planner_reward += output["cumulative_plan_reward"] 
        all_events_count += output["num_event_obs"]
        event_obs_pairs.extend(output["event_obs_pairs"])
        obs_per_event_list.extend(output["obs_per_event_list"])    
    max_rev_time_list = []
    avg_rev_time_list = []
    event_count = 0
    for event in tqdm(events):
        obs_per_event = []
        event_start = float(event[2])
        event_end = float(event[2])+float(event[3])
        for eop in event_obs_pairs:
            if eop["event"] == event:
                obs_per_event.append(eop["obs"])
        if len(obs_per_event) != 0:
            max_rev_time = compute_max_revisit_time(event_start,event_end,obs_per_event,settings)
            avg_rev_time = compute_avg_revisit_time(event_start,event_end,obs_per_event,settings)
            max_rev_time_list.append(max_rev_time)
            avg_rev_time_list.append(avg_rev_time)
            event_count += 1
    if len(events) > 0:
        events_perc_cov = event_count / len(events)
    else:
        events_perc_cov = 0.0

    all_max_rev_time_list = []
    all_avg_rev_time_list = []
    loc_count = 0
    for loc in tqdm(grid_locations):
        obs_per_loc = []
        for ob in obs:
            if close_enough(ob[2],ob[3],loc[0],loc[1]):
                obs_per_loc.append(ob)
        max_rev_time = compute_max_revisit_time(0,86400,obs_per_loc,settings)
        avg_rev_time = compute_avg_revisit_time(0,86400,obs_per_loc,settings)
        all_max_rev_time_list.append(max_rev_time)
        all_avg_rev_time_list.append(avg_rev_time)
        loc_count += 1
    locations_perc_cov = loc_count / len(grid_locations)

    print("Number of observations: "+str(len(obs)))
    print("Number of total events: "+str(len(events)))
    print("Number of event co-observations: "+str(all_events_count))
    print("Number of events observed at least once: "+str(np.count_nonzero(obs_per_event_list)))
    #print("Percent of events observed at least once: "+str(np.count_nonzero(obs_per_event_list)/len(events)*100)+"%")
    obs_per_event_array = np.array(obs_per_event_list)
    print("Average obs per event seen once: "+str(obs_per_event_array[np.nonzero(obs_per_event_array)].mean()))
    if len(max_rev_time_list) > 0:
        event_max_revisit_time = np.max(max_rev_time_list)
        event_avg_revisit_time = np.average(avg_rev_time_list)
    else:
        event_max_revisit_time = 86400 # TODO replace with simulation duration
        event_avg_revisit_time = 86400 # TODO replace with simulation duration
    results = {
        "event_obs_count": all_events_count,
        "events_seen_at_least_once": np.count_nonzero(obs_per_event_list),
        "events_seen_once": np.count_nonzero(np.asarray(obs_per_event_list) == 1),
        "events_seen_twice": np.count_nonzero(np.asarray(obs_per_event_list) == 2),
        "events_seen_thrice": np.count_nonzero(np.asarray(obs_per_event_list) == 3),
        "events_seen_fourplus": np.count_nonzero(np.asarray(obs_per_event_list) > 3),
        "events_seen_once_average": obs_per_event_array[np.nonzero(obs_per_event_array)].mean(),
        "obs_per_event_list": obs_per_event_list,
        "event_reward": event_reward,
        "planner_reward": planner_reward,
        "percent_coverage": events_perc_cov,
        "event_max_revisit_time": event_max_revisit_time, # max of max
        "event_avg_revisit_time": event_avg_revisit_time, # average of average
        "all_percent_coverage": locations_perc_cov,
        "all_max_revisit_time": np.max(all_max_rev_time_list), # max of max
        "all_avg_revisit_time": np.average(all_avg_rev_time_list) # average of average
    }
    return results

def compute_experiment_statistics(settings):
    directory = settings["directory"]+"orbit_data/"

    satellites = []
    all_initial_observations = []
    all_replan_observations = []
    all_oracle_observations = []
    #all_visibilities = []


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

            if "init" in f and settings["planner"] in f:
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
                all_initial_observations.extend(observations)

            if "replan" in f and settings["planner"] in f and "het" not in f and "oracle" not in f and "init" not in f:
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
                all_replan_observations.extend(observations)

            if "oracle" in f and settings["planner"] in f and "het" not in f:
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
                all_oracle_observations.extend(observations)
        if settings["preplanned_observations"] is not None:
            with open(settings["preplanned_observations"],newline='') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                observations = []
                i = 0
                for row in csvreader:
                    if i < 1:
                        i=i+1
                        continue
                    if int(row[0][8:]) == int(satellite["orbitpy_id"][3]):
                        obs = [int(float(row[3])),int(float(row[4])),float(row[1])*180/np.pi, float(row[2])*180/np.pi]
                        observations.append(obs)
            satellite["observations"] = observations
            

        if "orbitpy_id" in satellite:
            satellites.append(satellite)
    # all_visibilities = []
    # for satellite in satellites:
    #     vis_windows = []
    #     i = 0
    #     visibilities = satellite["visibilities"]
    #     while i < len(visibilities):
    #         continuous_visibilities = []
    #         visibility = visibilities[i]
    #         continuous_visibilities.append(visibility)
    #         start = visibility[0]
    #         end = visibility[0]
    #         while(i < len(visibilities)-1 and visibilities[i+1][0] == start):
    #             i += 1
    #         vis_done = False
    #         if i == len(visibilities)-1:
    #             break
    #         while not vis_done:
    #             vis_done = True
    #             num_steps = len(continuous_visibilities)
    #             while visibilities[i+1][0] == start+num_steps:
    #                 if visibilities[i+1][1] == visibility[1]:
    #                     continuous_visibilities.append(visibilities[i+1])
    #                     end = visibilities[i+1][0]
    #                     vis_done = False
    #                 if i == len(visibilities)-2:
    #                     break
    #                 else:
    #                     i += 1
    #             num_steps = len(continuous_visibilities)
    #             if i == len(visibilities)-1:
    #                 break
    #         vis_window = [start,end,visibility[3],visibility[4],0,0,visibility[-1]] # no reward or angle associated with visibilities
    #         vis_windows.append(vis_window)
    #         for cont_vis in continuous_visibilities:
    #             visibilities.remove(cont_vis)
    #         i = 0
    #     all_visibilities.extend(vis_windows)

    events = []
    event_filename = settings["event_csvs"][0]
    with open(event_filename,newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            events.append(row) # lat, lon, start, duration, severity
    
    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])

    print("Initial event observations")
    init_results = compute_statistics(events,all_initial_observations,grid_locations,settings)
    print("Replan event observations")
    replan_results = compute_statistics(events,all_replan_observations,grid_locations,settings)
    print("Oracle event observations")
    oracle_results = compute_statistics(events,all_oracle_observations,grid_locations,settings)
    # print("Potential observations (visibilities)")
    # vis_results = compute_statistics(events,all_visibilities,grid_locations,settings)
    overall_results = {
        "init_results": init_results,
        "replan_results": replan_results,
        "oracle_results": oracle_results,
        "num_events": len(events),
        "num_obs_init": len(all_initial_observations),
        "num_obs_replan": len(all_replan_observations),
        "num_obs_oracle": len(all_oracle_observations)
    }
    return overall_results

def main():
    name = "dqn_test_fov_step_fullstate"
    settings = {
        "name": name,
        "instrument": {
            "ffor": 60,
            "ffov": 0
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
            "duration": 1, # days
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
    overall_results = compute_experiment_statistics(settings)
    #print(overall_results)

if __name__ == "__main__":
    main()