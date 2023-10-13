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
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.01:
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
        start_list.append(obs[0]*settings["step_size"])
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
        start_list.append(obs[0]*settings["step_size"])
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
    satellite_name_dict = input["satellite_name_dict"]
    event_obs_pairs = []
    num_event_obs = 0
    cumulative_event_reward = 0
    cumulative_plan_reward = 0
    obs_per_event_list = []
    event_duration = settings["experiment_settings"]["event_duration"]
    ss = settings["step_size"]
    for event in events:
        measurements_str : str = event[5]
        measurements_str = measurements_str.replace('[','')
        measurements_str = measurements_str.replace(']','')
        measurements_str = measurements_str.replace(' ','')
        measurements_str = measurements_str.replace('\'','')
        measurements = measurements_str.split(',')
        obs_per_event = 0
        event_reward = float(event[4])
        obss = []
        for obs in observations:
            if obs[0] > ((float(event[2])/ss+float(event[3])/ss) + event_duration/ss):
                break
            if close_enough(obs[2],obs[3],float(event[0]),float(event[1])):
                if ((float(event[2])/ss) < obs[0] < (float(event[2])/ss+float(event[3])/ss)) or ((float(event[2])/ss) < obs[1] < (float(event[2])/ss+float(event[3])/ss)):
                    obss.append(obs)
                    obs_per_event += 1
                    cumulative_plan_reward += settings["experiment_settings"]["reward"]
                    if satellite_name_dict[obs[6]] in measurements:
                        measurements.remove(satellite_name_dict[obs[6]])

        if len(measurements) == 0:
            event_obss_pair = {
                "event": event,
                "obs": obss
            }
            event_obs_pairs.append(event_obss_pair)
            cumulative_event_reward += event_reward
            num_event_obs += 1
        obs_per_event_list.append(obs_per_event)


    output = {}
    output["event_obs_pairs"] = event_obs_pairs
    output["num_event_obs"] = num_event_obs
    output["obs_per_event_list"] = obs_per_event_list
    output["cumulative_event_reward"] = cumulative_event_reward
    output["cumulative_plan_reward"] = cumulative_plan_reward
    return output

def compute_statistics(events,obs,grid_locations,settings):
    satellite_name_dict = {}
    for i in range(settings["num_sats_per_plane"]*settings["num_planes"]*settings["experiment_settings"]["num_event_types"]**2):
        if i < settings["num_sats_per_plane"]*settings["num_planes"]*settings["experiment_settings"]["num_event_types"]:
            meas_type = "0"
        elif i < 2*settings["num_sats_per_plane"]*settings["num_planes"]*settings["experiment_settings"]["num_event_types"]:
            meas_type = "1"
        elif i < 3*settings["num_sats_per_plane"]*settings["num_planes"]*settings["experiment_settings"]["num_event_types"]:
            meas_type = "2"
        elif i < 4*settings["num_sats_per_plane"]*settings["num_planes"]*settings["experiment_settings"]["num_event_types"]:
            meas_type = "3"
        satellite_name_dict["sat"+str(i)] = meas_type
    obs.sort(key=lambda obs: obs[0])
    event_chunks = list(chunks(events,1))
    pool = multiprocessing.Pool()
    input_list = []
    for i in range(len(event_chunks)):
        input = {}
        input["events"] = event_chunks[i]
        input["observations"] = obs
        input["settings"] = settings
        input["satellite_name_dict"] = satellite_name_dict
        input_list.append(input)
    #output_list = pool.map(compute_statistics_pieces, input_list)
    output_list = list(tqdm(pool.imap(compute_statistics_pieces, input_list)))
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
                obs_per_event.extend(eop["obs"])
        if len(obs_per_event) != 0:
            max_rev_time = compute_max_revisit_time(event_start,event_end,obs_per_event,settings)
            avg_rev_time = compute_avg_revisit_time(event_start,event_end,obs_per_event,settings)
            max_rev_time_list.append(max_rev_time)
            avg_rev_time_list.append(avg_rev_time)
            event_count += 1
    events_perc_cov = event_count / len(events)

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

    print("Number of event co-observations: "+str(all_events_count))
    print("Number of total events: "+str(len(events)))
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
        "events_seen_once": np.count_nonzero(obs_per_event_list),
        "events_seen_once_average": obs_per_event_array[np.nonzero(obs_per_event_array)].mean(),
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

def compute_experiment_statistics_het(settings):
    directory = settings["directory"]+"orbit_data/"

    satellites = []
    all_initial_observations = []
    all_replan_observations = []
    all_visibilities = []


    for subdir in os.listdir(directory):
        satellite = {}
        if "comm" in subdir:
            continue
        if ".json" in subdir:
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

            if "plan" in f and not "replan" in f and settings["planner"] in f:
                with open(directory+subdir+"/"+f,newline='') as csv_file:
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
                all_initial_observations.extend(observations)

            if "replan" in f and settings["planner"] in f:
                with open(directory+subdir+"/"+f,newline='') as csv_file:
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
                all_replan_observations.extend(observations)

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

    all_visibilities = []
    for satellite in satellites:
        vis_windows = []
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
            vis_window = [start,end,visibility[3],visibility[4],0,0,visibility[-1]] # no reward or angle associated with visibilities
            vis_windows.append(vis_window)
            for cont_vis in continuous_visibilities:
                visibilities.remove(cont_vis)
            i = 0
        all_visibilities.extend(vis_windows)

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
    print("Potential observations (visibilities)")
    vis_results = compute_statistics(events,all_visibilities,grid_locations,settings)
    overall_results = {
        "init_results": init_results,
        "replan_results": replan_results,
        "vis_results": vis_results,
        "num_events": len(events),
        "num_obs_init": len(all_initial_observations),
        "num_obs_replan": len(all_replan_observations),
        "num_vis": len(all_visibilities)
    }
    print(overall_results)
    return overall_results

def main():
    experiment_settings = {
        "name": "oa_het_1",
        "ffor": 30,
        "ffov": 5,
        "constellation_size": 2,
        "agility": 1,
        "event_duration": 6*3600,
        "event_frequency": 0.01/3600,
        "event_density": 1,
        "event_clustering": 4,
        "planner": "dp",
        "reward": 10,
        "reobserve_reward": 2.0,
        "num_event_types": 4
    }
    mission_name = experiment_settings["name"]
    cross_track_ffor = experiment_settings["ffor"]
    along_track_ffor = experiment_settings["ffor"]
    cross_track_ffov = experiment_settings["ffov"]
    along_track_ffov = experiment_settings["ffov"] # TODO carefully consider this assumption
    agility = experiment_settings["agility"]
    num_planes = experiment_settings["constellation_size"]
    num_sats_per_plane = experiment_settings["constellation_size"]
    var = experiment_settings["event_clustering"]
    num_points_per_cell = experiment_settings["event_density"]
    event_frequency = experiment_settings["event_frequency"]
    event_duration = experiment_settings["event_duration"]
    num_event_types = experiment_settings["num_event_types"]
    simulation_step_size = 10 # seconds
    simulation_duration = 1 # days
    settings = {
        "directory": "./missions/"+mission_name+"/",
        "step_size": simulation_step_size,
        "duration": simulation_duration,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "point_grid": "./coverage_grids/"+mission_name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+mission_name+"/events.csv"],
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False,
        "planner": experiment_settings["planner"],
        "reward": experiment_settings["reward"],
        "experiment_settings": experiment_settings
    }
    overall_results = compute_experiment_statistics_het(settings)
    print(overall_results)

if __name__ == "__main__":
    main()