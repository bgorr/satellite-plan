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
from src.plan_mission import load_events, load_obs, load_rewards, load_satellites, chop_obs_list, close_enough
from src.utils.planning_utils import check_maneuver_feasibility
from src.utils.convert_geo import convert_geo_coords
from src.utils.compute_experiment_statistics import compute_experiment_statistics
from src.plan_mission_fov import plan_mission_replan_interval, plan_mission_horizon



from src.multiagent_rl.madqn_agent_fullstate import Agent
import matplotlib.pyplot as plt


# state space: satellite time, satellite angle
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
        with open("./missions/"+mission_name+"/events/events.csv",'w') as csvfile:
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

def transition_function(satellites, events, event_statuses, actions, num_actions, settings, grid_locations):
    observed_points = []
    new_state = []
    reward = 0
    done_flag = False
    for i, satellite in enumerate(satellites):
        planning_interval = 10
        if satellite["curr_time"]+planning_interval > settings["time"]["duration"]*86400/10:
            done_flag = True
            break
        obs_list = chop_obs_list(satellite["obs_list"],satellite["curr_time"],satellite["curr_time"]+planning_interval)
        pointing_options = np.arange(-settings["instrument"]["ffor"]/2,settings["instrument"]["ffor"]/2,settings["instrument"]["ffov"])
        pointing_option = pointing_options[actions[i]]
        slew_time = np.abs(satellite["curr_angle"]-pointing_option)/settings["agility"]["max_slew_rate"]/settings["time"]["step_size"]
        ready_time = satellite["curr_time"]+slew_time
        
        for obs in obs_list:
            if obs["end"] > ready_time and (np.abs(pointing_option-obs["angle"]) < settings["instrument"]["ffov"]/2):
                observed_points.append(obs)
        satellite["curr_time"] += planning_interval
        satellite["curr_angle"] = pointing_option
        satellite["curr_lat"] = satellite["ssps"][satellite["curr_time"]*10][0]
        satellite["curr_lon"] = satellite["ssps"][satellite["curr_time"]*10][1]
        new_state.append(satellite["curr_time"])
        new_state.append(satellite["curr_angle"])
        new_state.append(satellite["curr_lat"])
        new_state.append(satellite["curr_lon"])
    
    if done_flag:
        for grid_location in grid_locations:
            new_state.append(0)
        return new_state, 0, True, []
    event_locations = []
    not_event_locations = []
    for obs in observed_points:
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
            event_locations.append([location["lat"],location["lon"]])
            reward += 10
        else:
            not_event_locations.append([location["lat"],location["lon"]])
            reward += 1

    for i, grid_location in enumerate(grid_locations):
        event_occurring = False
        event_not_occurring = False
        for event_location in event_locations:
            if close_enough(event_location[0],event_location[1],grid_location[0],grid_location[1]):
                event_occurring = True
        for event_location in not_event_locations:
            if close_enough(event_location[0],event_location[1],grid_location[0],grid_location[1]):
                event_not_occurring = True

        if event_occurring:
            new_state.append(1)
            print('heyo')
        elif event_statuses[i] == 1 and event_not_occurring:
            new_state.append(0)
        elif event_statuses[i] == 1 and not event_not_occurring:
            new_state.append(1)
            print('heyo')
        else:
            new_state.append(0)
    
    return new_state, reward, False, observed_points

def transition_function_by_sat(satellites, events, event_statuses, actions, num_actions, settings, grid_locations):
    observed_points = []
    observed_points_flattened = []
    new_state = []
    reward = 0
    done_flag = False
    for i, satellite in enumerate(satellites):
        observed_points_per_sat = []
        planning_interval = 10
        if satellite["curr_time"]+planning_interval > settings["time"]["duration"]*86400/10:
            done_flag = True
            break
        obs_list = chop_obs_list(satellite["obs_list"],satellite["curr_time"],satellite["curr_time"]+planning_interval)
        pointing_options = np.arange(-settings["instrument"]["ffor"]/2,settings["instrument"]["ffor"]/2,settings["instrument"]["ffov"])
        pointing_option = pointing_options[actions[i]]
        slew_time = np.abs(satellite["curr_angle"]-pointing_option)/settings["agility"]["max_slew_rate"]/settings["time"]["step_size"]
        ready_time = satellite["curr_time"]+slew_time
        
        for obs in obs_list:
            if obs["end"] > ready_time and (np.abs(pointing_option-obs["angle"]) < settings["instrument"]["ffov"]/2):
                observed_points_per_sat.append(obs)
                observed_points_flattened.append(obs)
        satellite["curr_time"] += planning_interval
        satellite["curr_angle"] = pointing_option
        satellite["curr_lat"] = satellite["ssps"][satellite["curr_time"]*10][0]
        satellite["curr_lon"] = satellite["ssps"][satellite["curr_time"]*10][1]
        new_state.append(satellite["curr_time"])
        new_state.append(satellite["curr_angle"])
        new_state.append(satellite["curr_lat"])
        new_state.append(satellite["curr_lon"])
        observed_points.append(observed_points_per_sat)
    
    if done_flag:
        for grid_location in grid_locations:
            new_state.append(0)
        return new_state, 0, True, []
    
    event_locations = []
    not_event_locations = []
    for obs in observed_points:
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
            event_locations.append([location["lat"],location["lon"]])
            reward += 10
        else:
            not_event_locations.append([location["lat"],location["lon"]])
            reward += 1

    for i, grid_location in enumerate(grid_locations):
        event_occurring = False
        event_not_occurring = False
        for event_location in event_locations:
            if close_enough(event_location[0],event_location[1],grid_location[0],grid_location[1]):
                event_occurring = True
        for event_location in not_event_locations:
            if close_enough(event_location[0],event_location[1],grid_location[0],grid_location[1]):
                event_not_occurring = True

        if event_occurring:
            new_state.append(1)
        elif event_statuses[i] == 1 and event_not_occurring:
            new_state.append(0)
        elif event_statuses[i] == 1 and not event_not_occurring:
            new_state.append(1)
        else:
            new_state.append(0)
    
    return new_state, reward, False, observed_points

if __name__ == '__main__':
    name = "madqn_test_fov_step_fullstate"
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
        "planner": "madqn",
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

    grid_locations = []
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])

    agent_list = []
    N = 100
    n_games = 1000
    n_steps = 0
    learn_iters = 0
    best_score = -1000
    figure_file = 'plots/madqn_fov_step_fullstate.png'
    score_history = []
    batch_size = 10
    n_epochs = 10
    alpha=0.00005
    action_space_size = len(np.arange(-settings["instrument"]["ffor"]/2,settings["instrument"]["ffor"]/2,settings["instrument"]["ffov"]))
    observation_space_size = 4*len(satellites) + len(grid_locations) + 1
    agent = Agent(settings=settings,n_sats=len(satellites),gamma=0.99, epsilon = 0.99, batch_size=256, n_actions=action_space_size, eps_end=0.01, input_dims=[observation_space_size], lr=0.00005)
    randomize_events = True
    for j in range(n_games):
        if randomize_events:
            events = create_and_load_random_events(settings)
        joint_observation = []
        for satellite in satellites:
            satellite["curr_angle"] = 0.0
            satellite["curr_time"] = 0.0
            satellite["curr_lat"] = satellite["ssps"][0.0][0]
            satellite["curr_lon"] = satellite["ssps"][0.0][1]
            joint_observation.append(satellite["curr_time"])
            joint_observation.append(satellite["curr_angle"])
            joint_observation.append(satellite["curr_lat"])
            joint_observation.append(satellite["curr_lon"])

        for grid_location in grid_locations:
            joint_observation.append(0)

        done = False
        score = 0
        while not done:
            actions = []
            for i, satellite in enumerate(satellites):
                joint_obs_w_sat_index = joint_observation + [np.float32(i)]
                action = agent.choose_action(joint_obs_w_sat_index)
                actions.append(action)
            joint_observation_, reward, done, obs_info = transition_function(satellites,events,joint_observation[4*len(satellites):],actions,action_space_size,settings,grid_locations)
            if done:
                break
            for i, satellite in enumerate(satellites):
                satellite["curr_time"] = joint_observation_[4*i]
                satellite["curr_angle"] = joint_observation_[4*i+1]
                satellite["curr_lat"] = joint_observation_[4*i+2]
                satellite["curr_lon"] = joint_observation_[4*i+3]
            n_steps += 1
            score += reward
            for i, satellite in enumerate(satellites):
                joint_obs_w_sat_index = joint_observation + [np.float32(i)]
                joint_obs_w_sat_index_ = joint_observation_ + [np.float32(i)]
                agent.store_transition(joint_obs_w_sat_index, actions[i], reward, joint_obs_w_sat_index_, done)
                agent.learn()
            joint_observation = joint_observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode', j, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
        
    ### LOADING SAVED MODELS AND SAVING PLANS ###

    agent = Agent(settings=settings,n_sats=len(satellites),gamma=0.99, epsilon = 0.0, batch_size=256, n_actions=action_space_size, eps_end=0.01, input_dims=[observation_space_size], lr=0.00005)
    agent.load_models()
    joint_observation = []
    plans = []
    for i in range(len(satellites)):
        plans.append([])
    for satellite in satellites:
        satellite["curr_angle"] = 0.0
        satellite["curr_time"] = 0.0
        satellite["curr_lat"] = satellite["ssps"][0.0][0]
        satellite["curr_lon"] = satellite["ssps"][0.0][1]
        joint_observation.append(satellite["curr_time"])
        joint_observation.append(satellite["curr_angle"])
        joint_observation.append(satellite["curr_lat"])
        joint_observation.append(satellite["curr_lon"])

    for grid_location in grid_locations:
        joint_observation.append(0)
    done = False
    score = 0
    while not done:
        actions = []
        for i, satellite in enumerate(satellites):
            joint_obs_w_sat_index = joint_observation + [i]
            action = agent.choose_action(joint_obs_w_sat_index)
            actions.append(action)
        joint_observation_, reward, done, obs_info = transition_function_by_sat(satellites,fixed_events,joint_observation[4*len(satellites):],actions,action_space_size,settings,grid_locations)
        if done:
            break
        for i, satellite in enumerate(satellites):
            satellite["curr_time"] = joint_observation_[4*i]
            satellite["curr_angle"] = joint_observation_[4*i+1]
            satellite["curr_lat"] = joint_observation_[4*i+2]
            satellite["curr_lon"] = joint_observation_[4*i+3]
            for obs in obs_info[i]:
                plans[i].append(obs)
        score += reward
        joint_observation = joint_observation_
    for i, satellite in enumerate(satellites):
        satellite["plan"] = plans[i]
        save_plan(satellite,settings,"")

    ### COMPUTING EXPERIMENT STATISTICS ###
    settings["event_csvs"] = ["./missions/"+name+"/events/events.csv"] 
    compute_experiment_statistics(settings)
    settings["planner"] = "dp"
    if not os.path.exists("./missions/"+settings["name"]+"/orbit_data/sat0/replan_intervaldphom.csv"):
        plan_mission_horizon(settings)
        plan_mission_replan_interval(settings)
    compute_experiment_statistics(settings)

