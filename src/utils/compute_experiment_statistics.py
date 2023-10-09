import numpy as np
import datetime
import csv
import os
import multiprocessing
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from planners.BaseRL import BaseRL
import config

from planners import utils


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

def compute_statistics_pieces(input):
    events = input["events"]
    observations = input["observations"]
    settings = input["settings"]
    event_obs_pairs = []
    num_event_obs = 0
    obs_per_event_list = []
    event_duration = settings["experiment_settings"]["event_duration"]
    ss = settings["step_size"]
    for event in events:
        obs_per_event = 0
        for obs in observations:
            if obs[0] > ((float(event[2])/ss+float(event[3])/ss) + event_duration/ss):
                break
            if close_enough(obs[2],obs[3],float(event[0]),float(event[1])):
                if ((float(event[2])/ss) < obs[0] < (float(event[2])/ss+float(event[3])/ss)) or ((float(event[2])/ss) < obs[1] < (float(event[2])/ss+float(event[3])/ss)):
                    event_obs_pair = {
                        "event": event,
                        "obs": obs
                    }
                    event_obs_pairs.append(event_obs_pair)
                    obs_per_event += 1
                    num_event_obs += 1
        obs_per_event_list.append(obs_per_event)

    output = {}
    output["event_obs_pairs"] = event_obs_pairs
    output["num_event_obs"] = num_event_obs
    output["obs_per_event_list"] = obs_per_event_list
    return output

def compute_statistics(events,obs,settings):
    obs.sort(key=lambda obs: obs[0])
    event_chunks = list(chunks(events, 10))

    input_list = [
        {'events': event_chunk, 'observations': obs, 'settings': settings} for event_chunk in event_chunks
    ]
    with Pool(processes=config.cores) as pool:
        output_list = list(tqdm(pool.imap(compute_statistics_pieces, input_list), total=len(input_list)))


    all_events_count = 0
    event_obs_pairs = []
    obs_per_event_list = []
    for output in output_list:
        all_events_count += output["num_event_obs"]
        event_obs_pairs.extend(output["event_obs_pairs"])
        obs_per_event_list.extend(output["obs_per_event_list"])

    print("Number of event observations: "+str(all_events_count))
    print("Number of total events: "+str(len(events)))
    print("Number of events observed at least once: "+str(np.count_nonzero(obs_per_event_list)))
    #print("Percent of events observed at least once: "+str(np.count_nonzero(obs_per_event_list)/len(events)*100)+"%")
    obs_per_event_array = np.array(obs_per_event_list)
    print("Average obs per event seen once: "+str(obs_per_event_array[np.nonzero(obs_per_event_array)].mean()))
    results = {
        "event_obs_count": all_events_count,
        "events_seen_once": np.count_nonzero(obs_per_event_list),
        "events_seen_once_average": obs_per_event_array[np.nonzero(obs_per_event_array)].mean()
    }
    return results

def compute_experiment_statistics(settings):
    directory = settings["directory"]+"orbit_data/"

    satellites = []
    all_initial_observations = []
    all_replan_observations = []

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
            vis_window = [start,end,visibility[3],visibility[4],visibility[-1]]
            vis_windows.append(vis_window)
            for cont_vis in continuous_visibilities:
                visibilities.remove(cont_vis)
            i = 0
        all_visibilities.extend(vis_windows)

    events = []
    event_filename = settings["event_csvs"][0]
    with open(event_filename, newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            events.append(row) # lat, lon, start, duration, severity

    print("Initial event observations")
    init_results = compute_statistics(events,all_initial_observations,settings)
    print("Replan event observations")
    replan_results = compute_statistics(events,all_replan_observations,settings)
    print("Potential observations (visibilities)")
    vis_results = compute_statistics(events,all_visibilities,settings)
    overall_results = {
        "init_results": init_results,
        "replan_results": replan_results,
        "vis_results": vis_results,
        "num_events": len(events),
        "num_obs_init": len(all_initial_observations),
        "num_obs_replan": len(all_replan_observations),
        "num_vis": len(all_visibilities)
    }
    return overall_results

def main():
    experiment_settings = {
        "name": "experiment_num_7",
        "ffor": 60,
        "ffov": 5,
        "constellation_size": 6,
        "agility": 1,
        "event_duration": 1.5*3600,
        "event_frequency": 0.01/3600,
        "event_density": 10,
        "event_clustering": 4,
        "planner": "heuristic",
        "planner_options": {
                "reobserve": "encouraged",
                "reobserve_reward": 2
        }
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
        "planner_options": experiment_settings["planner_options"],
        "experiment_settings": experiment_settings
    }
    compute_experiment_statistics(settings)

if __name__ == "__main__":
    main()