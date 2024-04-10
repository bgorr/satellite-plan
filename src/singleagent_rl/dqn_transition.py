"""Main PPO runner"""
import datetime
import os
import csv
import sys
import random
import numpy as np
sys.path.append(".")

from src.create_mission import create_mission
from src.execute_mission import execute_mission
from src.plan_mission import load_events, load_obs, load_rewards, load_satellites, chop_obs_list, close_enough, save_plan_w_fov
from src.utils.planning_utils import check_maneuver_feasibility
from src.utils.convert_geo import convert_geo_coords
from src.utils.compute_experiment_statistics import compute_experiment_statistics
from src.plan_mission_fov import plan_mission_replan_interval, plan_mission_horizon


from dqn_agent import Agent
import matplotlib.pyplot as plt

# action space: next 5 actions

def save_plan(satellite,settings,flag):
    directory = settings["directory"] + "orbit_data/"
    with open(directory+satellite["orbitpy_id"]+'/replan_interval'+settings["planner"]+flag+'.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        plan = satellite["plan"]
        rows = []
        for obs in plan:
            row = [obs["start"],obs["end"],obs["location"]["lat"],obs["location"]["lon"],obs["angle"],obs["reward"]]
            rows.append(row)

        for row in rows:
            csvwriter.writerow(row)

def create_events(settings):
    simulation_duration = settings["time"]["duration"] # days
    mission_name = settings["name"]
    event_duration = settings["events"]["event_duration"]
    if not os.path.exists("./missions/"+mission_name+"/coverage_grids/event_locations.csv"):
        possible_event_locations = []
        if settings["events"]["event_clustering"] == "uniform":
            center_lats = np.arange(-90,90,0.1)
            center_lons = np.arange(-180,180,0.1)
            for clat in center_lats:
                for clon in center_lons:
                    location = [clat,clon]
                    possible_event_locations.append(location)
        elif settings["events"]["event_clustering"] == "clustered":
            center_lats = np.random.uniform(-90,90,100)
            center_lons = np.random.uniform(-180,180,100)
            for i, _ in enumerate(center_lons):
                var = 1
                mean = [center_lats[i], center_lons[i]]
                cov = [[var, 0], [0, var]]
                num_points_per_cell = int(6.48e6/100)
                xs, ys = np.random.multivariate_normal(mean, cov, num_points_per_cell).T
                for i, _ in enumerate(xs):
                    location = [xs[i],ys[i]]
                    possible_event_locations.append(location)
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists("./missions/"+mission_name+"/events/"):
        events = []
        used_event_locations = []
        for i in range(settings["events"]["num_events"]):
            event_location = random.choice(possible_event_locations)
            used_event_locations.append(event_location)
            step = int(np.random.uniform(0,simulation_duration*86400))
            event = [event_location[0],event_location[1],step,event_duration,1]
            events.append(event)
        if not os.path.exists("./missions/"+mission_name+"/events/"):
            os.mkdir("./missions/"+mission_name+"/events/")
        with open("./missions/"+mission_name+"/events/events.csv",'w',encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
            for event in events:
                csvwriter.writerow(event)
        unused_event_locations = []
        for i in range(settings["events"]["num_event_locations"]-settings["events"]["num_events"]):
            unused_event_locations.append(random.choice(possible_event_locations))
    if not os.path.exists("./missions/"+mission_name+"/coverage_grids/"):
        os.mkdir("./missions/"+mission_name+"/coverage_grids/")
        with open("./missions/"+mission_name+"/coverage_grids/event_locations.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]'])
            for location in used_event_locations:
                csvwriter.writerow(location)
            for location in unused_event_locations:
                csvwriter.writerow(location)


def create_and_load_random_events(settings):
    simulation_duration = settings["time"]["duration"] # days
    mission_name = settings["name"]
    event_duration = settings["events"]["event_duration"]
    possible_event_locations = []
    with open("./missions/"+mission_name+"/coverage_grids/event_locations.csv","r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            possible_event_locations.append([float(row[0]),float(row[1])])
    events = []
    for i in range(settings["events"]["num_events"]):
        event_location = random.choice(possible_event_locations)
        step = int(np.random.uniform(0,simulation_duration*86400))
        event = [event_location[0],event_location[1],step,event_duration,1]
        events.append(event)
    with open("./missions/"+mission_name+"/events/random_events.csv",'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
        for event in events:
            csvwriter.writerow(event)
    settings["event_csvs"] = ["./missions/"+mission_name+"/events/random_events.csv"]
    return load_events(settings)

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i, _ in enumerate(running_avg):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def transition_function(satellite, events, action, num_actions, settings):
    obs_list = chop_obs_list(satellite["obs_list"],satellite["curr_time"],8640)
    feasible_actions = []
    idx = 0
    while len(feasible_actions) < num_actions and idx < len(obs_list):
        feasible, _ = check_maneuver_feasibility(satellite["curr_angle"], obs_list[idx]["angle"], satellite["curr_time"], obs_list[idx]["end"], settings)
        if feasible:
            feasible_actions.append(obs_list[idx])
        idx += 1
    if action >= len(feasible_actions):
        action = len(feasible_actions)-1
    if len(feasible_actions) == 0:
        satellite["curr_lat"] = satellite["ssps"][satellite["curr_time"]*10][0]
        satellite["curr_lon"] = satellite["ssps"][satellite["curr_time"]*10][1]
        return [satellite["curr_time"],satellite["curr_angle"],satellite["curr_lat"],satellite["curr_lon"]], 0, True, None
    obs = feasible_actions[action]
    location = obs["location"]
    events_per_location = []
    for event in events:
        if close_enough(location["lat"],location["lon"],event["location"]["lat"],event["location"]["lon"]):
            events_per_location.append(event)
    event_occurring = False
    for event in events_per_location:
        if event["start"] <= obs["start"] <= event["end"]:
            event_occurring = True
    if event_occurring:
        reward = 10
    else:
        reward = 1
    satellite["curr_lat"] = satellite["ssps"][obs["end"]*10][0]
    satellite["curr_lon"] = satellite["ssps"][obs["end"]*10][1]

    return [obs["end"],obs["angle"],satellite["curr_lat"],satellite["curr_lon"]], reward, False, obs

if __name__ == '__main__':
    name = "dqn_transition"
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
            "num_sats_per_plane": 5,
            "num_planes": 1,
            "phasing_parameter": 1
        },
        "events": {
            "event_duration": 3600*6,
            "num_event_locations": 10000,
            "num_events": 1000,
            "event_clustering": "clustered"
        },
        "time": {
            "step_size": 10, # seconds
            "duration": 0.1, # days
            "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
        },
        "rewards": {
            "reward": 10,
            "reward_increment": 2,
            "reobserve_conops": "no_change",
            "event_duration_decay": "step",
            "no_event_reward": 1,
            "oracle_reobs": "true",
            "initial_reward": 1
        },
        "plotting":{
            "plot_clouds": False,
            "plot_rain": False,
            "plot_duration": 0.1,
            "plot_interval": 10,
            "plot_obs": True
        },
        "planner": "dqn",
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
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
    create_events(settings)
    create_mission(settings)
    execute_mission(settings)
    convert_geo_coords(settings)
    directory = settings["directory"] + "orbit_data/"
    if "point_grid" not in settings:
        settings["point_grid"] = settings["directory"]+"orbit_data/grid0.csv"
    satellites = load_satellites(directory)
    events = load_events(settings)
    fixed_events = events
    rewards = load_rewards(settings)
    for satellite in satellites:
        satellite["obs_list"] = load_obs(satellite)
    randomize_events = False
    combined_score = 0
    for satellite in satellites:
        N = 100
        batch_size = 10
        n_epochs = 10
        alpha=0.00005
        action_space_size = 5
        observation_space_size = 4
        agent = Agent(settings, satellite["orbitpy_id"],gamma=0.99, epsilon = 0.99, batch_size=256, n_actions=action_space_size, eps_end=0.01, input_dims=[observation_space_size], lr=0.00005)
        n_games = 10000

        figure_file = 'plots/dqn_transition_'+satellite["orbitpy_id"]+'.png'
        best_score = -1000
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0
        for i in range(n_games):
            if randomize_events:
                events = create_and_load_random_events(settings)
            satellite["curr_angle"] = 0.0
            satellite["curr_time"] = 0.0
            satellite["curr_lat"] = satellite["ssps"][0.0][0]
            satellite["curr_lon"] = satellite["ssps"][0.0][1]
            observation = [satellite["curr_time"],satellite["curr_angle"],satellite["curr_lat"],satellite["curr_lon"]]
            done = False
            score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, obs_info = transition_function(satellite,events,action,action_space_size,settings)
                satellite["curr_time"] = observation_[0]
                satellite["curr_angle"] = observation_[1]
                satellite["curr_lat"] = observation_[2]
                satellite["curr_lon"] = observation_[3]
                n_steps += 1
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
        combined_score += best_score
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
    print(combined_score)

    ### LOADING SAVED MODELS AND SAVING PLANS ###
    for satellite in satellites:
        N = 100
        batch_size = 10
        n_epochs = 10
        alpha=0.00005
        action_space_size = 5
        observation_space_size = 4
        agent = Agent(settings, sat_name=satellite["orbitpy_id"],gamma=0.99, epsilon = 0.0, batch_size=256, n_actions=action_space_size, eps_end=0.01, input_dims=[observation_space_size], lr=0.00005)
        agent.load_models()
        plan = []
        satellite["curr_angle"] = 0.0
        satellite["curr_time"] = 0.0
        satellite["curr_lat"] = satellite["ssps"][0.0][0]
        satellite["curr_lon"] = satellite["ssps"][0.0][1]
        observation = [satellite["curr_time"],satellite["curr_angle"],satellite["curr_lat"],satellite["curr_lon"]]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, obs_info = transition_function(satellite,fixed_events,action,action_space_size,settings)
            plan.append(obs_info)
            observation = observation_
        satellite["plan"] = plan
        save_plan(satellite,settings,"")

    ### COMPUTING EXPERIMENT STATISTICS ###
    settings["event_csvs"] = ["./missions/"+name+"/events/events.csv"] 
    compute_experiment_statistics(settings)
    settings["planner"] = "dp"
    if not os.path.exists("./missions/"+settings["name"]+"/orbit_data/sat0/replan_intervaldphom.csv"):
        plan_mission_horizon(settings)
        plan_mission_replan_interval(settings)
    compute_experiment_statistics(settings)
