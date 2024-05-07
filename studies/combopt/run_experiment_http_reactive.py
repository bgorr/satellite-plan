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
from src.plan_mission_fov import plan_mission_replan_interval
from src.utils.complete_plan import complete_plan
from src.utils.compute_experiment_statistics import unique, close_enough, chunks, compute_statistics_pieces
from flask import Flask, request

def spectrometer_model(dist,settings):
    alt = settings["orbit"]["altitude"]
    focal_length = settings["spectrometer_specs"]["focal_length"]
    aperture = settings["spectrometer_specs"]["aperture"]
    pixel_size = settings["spectrometer_specs"]["pixel_size"]
    spectral_resolution = (1000-380)/settings["spectrometer_specs"]["spectral_pixels"]
    max_wavelength = 1000e-9
    orbital_velocity = np.sqrt(398600/(6378 + alt))
    ground_velocity = orbital_velocity * 6378 / (6378 + alt)
    ifov = pixel_size / focal_length
    gsd = ifov * dist * 1000
    diffraction_limited_resolution = 1.22 * dist * 1000 * max_wavelength / aperture
    spatial_resolution = np.max([diffraction_limited_resolution, gsd])

    imaging_rate = ground_velocity * 1000 / gsd

    vnir_L = 0.1 # approximate, at 600 nm
    vnir_lambda = 600e-9
    c = 3e8
    h = 6.63e-34
    eff = 0.8
    vnir_signal = vnir_lambda*vnir_L*np.pi*np.power(aperture,2)*np.power(pixel_size,2)*(1/imaging_rate)*eff*spectral_resolution/(4*h*c*np.power(focal_length,2))
    vnir_noise = np.sqrt(np.power(np.sqrt(vnir_signal),2)+10000)
    snr = vnir_signal / vnir_noise

    return spatial_resolution, spectral_resolution, snr

def science_model_molly(spatial, spectral, snr):
    beta = 0.5
    gamma = 0.5
    a_h = 9.94e-2
    b_h = 7.98e-5
    c_h = 1.78e-3
    d_h = 0.711
    a_i = 1.97e-6
    b_i = -7e-3
    c_i = 13.14
    H = a_h * np.log(snr) - b_h * spatial - c_h * spectral + d_h
    I = a_i * np.exp(b_i * spatial + c_i)
    science_value = beta*H + gamma*I # omitting data quantity
    return science_value

def get_observation_reward(obs,settings):
    dist = settings["orbit"]["altitude"] / np.cos(np.deg2rad(obs["angle"]))
    spatial, spectral, snr = spectrometer_model(dist,settings)
    multiplier = science_model_molly(spatial, spectral, snr)
    return multiplier

def compute_statistics(events,obs,grid_locations,settings):
    obs.sort(key=lambda obs: obs[0])
    event_chunks = list(chunks(events,25))
    output_list = []
    for i in range(len(event_chunks)):
        input = dict()
        input["events"] = event_chunks[i]
        input["observations"] = obs
        input["settings"] = settings
        output_list.append(compute_statistics_pieces(input))
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
    reward = 0
    for eop in event_obs_pairs:
        obs = eop["obs"]
        obs_dict = {
            "start": obs[0],
            "end": obs[1],
            "lat": obs[2],
            "lon": obs[3],
            "angle": obs[4],
            "reward": obs[5],
            "sat": obs[6]
        }
        reward += get_observation_reward(obs_dict,settings)
    return reward

def compute_experiment_reward(settings):
    directory = settings["directory"]+"orbit_data/"

    satellites = []
    all_initial_observations = []
    all_replan_observations = []
    all_oracle_observations = []

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

    reward = compute_statistics(events,all_replan_observations,grid_locations,settings)
    return reward

def delete_mission(settings):
    dirpath = settings["directory"]
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

def run_experiment(settings):  
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
        create_mission(settings)
        execute_mission(settings)
    average_reward = 0
    for i in range(3):
        settings["event_csvs"] = ["./events/http_events/events"+str(i)+".csv"]
        #plan_mission_horizon(settings) # must come before process as process expects a plan.csv in the orbit_data directory
        plan_mission_replan_interval(settings)
        complete_plan("hom",settings)
        average_reward += compute_experiment_reward(settings)
        #overall_results = compute_experiment_statistics(settings)
        #average_reward += overall_results["replan_results"]["event_obs_count"]
    average_reward = average_reward / 3
    delete_mission(settings)
    return average_reward


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
        spectral_pixels = int(request.form.get("spectral_pixels"))
        focal_length = float(request.form.get("focal_length"))
        aperture = float(request.form.get("aperture"))
        pixel_size = float(request.form.get("pixel_size"))
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
            "spectrometer_specs": {
                "spectral_pixels": spectral_pixels,
                "focal_length": focal_length,
                "aperture": aperture,
                "pixel_size": pixel_size
            },
            "planner": "dp",
            "num_meas_types": 3,
            "sharing_horizon": 500,
            "planning_horizon": 500,
            "directory": "./missions/"+name+"/",
            "grid_type": "custom", # can be "uniform" or "custom"
            "point_grid": "./coverage_grids/http_events/event_locations.csv",
            "preplanned_observations": None,
            "event_csvs": ["./events/"+name+"/events.csv"],
            "process_obs_only": False,
            "conops": "onboard_processing"
        }
        reward = run_experiment(settings)
        #reward = overall_results["replan_results"]["event_obs_count"]
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