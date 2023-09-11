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

def compute_statistics_pieces(input):
    events = input["events"]
    observations = input["observations"]
    event_obs_pairs = []
    num_event_obs = 0
    obs_per_event_list = []
    step_size = 10
    event_duration = 7200

    for event in events:
        obs_per_event = 0
        for obs in observations:
            if obs[0]*step_size > ((float(event[2])+float(event[3])) + event_duration):
                break
            if close_enough(obs[2],obs[3],float(event[0]),float(event[1])):
                if (float(event[2]) < obs[0]*step_size < (float(event[2])+float(event[3]))) or (float(event[2]) < obs[1]*step_size < (float(event[2])+float(event[3]))):
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

def compute_statistics(events,obs):
    obs.sort(key=lambda obs: obs[0])
    print(len(obs))
    event_chunks = list(chunks(events,1))
    pool = multiprocessing.Pool()
    input_list = []
    for i in range(len(event_chunks)):
        input = {}
        input["events"] = event_chunks[i]
        input["observations"] = obs
        input_list.append(input)
    print(len(input_list))
    #output_list = pool.map(compute_statistics_pieces, input_list)
    output_list = list(tqdm(pool.imap(compute_statistics_pieces, input_list)))
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
    print("Percent of events observed at least once: "+str(np.count_nonzero(obs_per_event_list)/len(events)*100)+"%")
    print("Average obs per event: "+str(np.average(obs_per_event_list)))

mission_name = "experiment1"
cross_track_ffor = 60 # deg
along_track_ffor = 60 # deg
cross_track_ffov = 10 # deg
along_track_ffov = 10 # deg
agility = 1 # deg/s
num_planes = 5 
num_sats_per_plane = 5
var = 10 # deg lat/lon
num_points_per_cell = 20
simulation_step_size = 10 # seconds
simulation_duration = 1 # days
event_frequency = 1e-4 # events per second
event_duration = 7200 # second
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
    "plot_duration": 1,
    "plot_interval": 20,
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

directory = settings["directory"]+"orbit_data/"

satellites = []
all_observations = []
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

        if "plan.csv" == f:
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
            satellite["observations"] = observations
            all_observations.extend(observations)

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
        start = visibility[0]*settings["step_size"]
        end = visibility[0]*settings["step_size"]
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
                    end = visibilities[i+1][0]*settings["step_size"]
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
    print(len(vis_windows))
    all_visibilities.extend(vis_windows)
print(len(all_visibilities))

events = []
event_filename = './events/experiment1/events.csv'
with open(event_filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 1:
            i=i+1
            continue
        events.append(row) # lat, lon, start, duration, severity

print("Actual observations")
compute_statistics(events,all_observations)
print("Potential observations (visibilities)")
compute_statistics(events,all_visibilities)
