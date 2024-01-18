from utils.planning_utils import get_action_space, close_enough

def fifo_planner(obs_list,settings):
    fifo_plan = []
    curr_time = 0.0
    curr_angle = 0.0
    last_obs = None
    while True:
        actions = get_action_space(curr_time,curr_angle,obs_list,last_obs,settings)
        if len(actions) == 0:
            break
        next_obs = actions[0]
        fifo_plan.append(next_obs)
        curr_time = next_obs["soonest"]
        curr_angle = next_obs["angle"]
    return fifo_plan

def fifo_planner_events(planner_inputs):
    events = planner_inputs["events"]
    obs_list = planner_inputs["obs_list"]
    plan_start = planner_inputs["plan_start"]
    plan_end = planner_inputs["plan_end"]
    settings = planner_inputs["settings"]
    fifo_plan = []
    curr_time = plan_start
    curr_angle = 0.0
    last_obs = None
    while True:
        actions = get_action_space(curr_time,curr_angle,obs_list,last_obs,settings)
        if len(actions) == 0:
            break
        next_obs = actions[0]
        curr_time = next_obs["soonest"]
        curr_angle = next_obs["angle"]
        fifo_plan.append(next_obs)
        for event in events:
            if close_enough(next_obs["location"]["lat"],next_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                if (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"]):
                    updated_reward = { 
                        "reward": next_obs["reward"],
                        "location": next_obs["location"],
                        "last_updated": curr_time 
                    }
                    planner_outputs = {
                        "plan": fifo_plan,
                        "end_time": curr_time,
                        "updated_reward": updated_reward
                    }
                    return planner_outputs
    planner_outputs = {
        "plan": fifo_plan,
        "end_time": plan_end,
        "updated_reward": None
    }  
    return planner_outputs

def fifo_planner_events_interval(planner_inputs):
    events = planner_inputs["events"]
    obs_list = planner_inputs["obs_list"].copy()
    plan_start = planner_inputs["plan_start"]
    plan_end = planner_inputs["plan_end"]
    settings = planner_inputs["settings"]
    fifo_plan = []
    curr_time = plan_start
    curr_angle = 0.0
    last_obs = None
    updated_rewards = []
    while True:
        actions = get_action_space(curr_time,curr_angle,obs_list,last_obs,settings)
        if len(actions) == 0:
            break
        next_obs = actions[0]
        obs_list.remove(next_obs)
        curr_time = next_obs["soonest"]
        curr_angle = next_obs["angle"]
        fifo_plan.append(next_obs)
        for event in events:
            if close_enough(next_obs["location"]["lat"],next_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                if (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"]):
                    updated_reward = { 
                            "reward": event["severity"]*settings["experiment_settings"]["reward"],
                            "location": next_obs["location"],
                            "last_updated": curr_time 
                        }
                    updated_rewards.append(updated_reward)
        if curr_time > plan_end:
            break
    planner_outputs = {
        "plan": fifo_plan,
        "end_time": plan_end,
        "updated_rewards": updated_rewards
    }  
    return planner_outputs