import numpy as np
from src.utils.planning_utils import get_action_space, close_enough

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
                duration = 86400/settings["time"]["step_size"]
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
    settings = planner_inputs["settings"]
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
            actions = get_action_space(curr_time,curr_angle,obs_list,last_obs,settings)
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
                    if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                        updated_reward = { 
                            "reward": event["severity"]*settings["rewards"]["reward"],
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
    settings = planner_inputs["settings"]
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
            actions = get_action_space(curr_time,curr_angle,obs_list,last_obs,settings)
            if(len(actions) == 0):
                break
            for action in actions:
                if action["location"] in location_list and action["reward"] == 1:
                    action["reward"] = settings["rewards"]["reobserve_reward"]
                duration = 86400/settings["time"]["step_size"] # TODO FIX
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
                    if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                        updated_reward = { 
                            "reward": event["severity"]*settings["rewards"]["reward"],
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