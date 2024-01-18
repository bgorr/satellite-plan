import numpy as np
import random
from utils.planning_utils import close_enough, get_action_space

class monte_carlo_tree_search():
    def __init__(self):
        self.V = []
        self.NQ = {}

    def do_search(self,obs_list,settings):
        self.obs_list = obs_list
        new_obs_list = []
        for obs in self.obs_list:
            new_obs = {
                "start": obs["start"],
                "end": obs["end"],
                "angle": obs["angle"],
                "reward": obs["reward"],
                "location": tuple(sorted(obs["location"].items()))
            }
            new_obs_list.append(new_obs)
        self.obs_list = new_obs_list
        self.sim_settings = settings
        result_list = []
        initial_state = {
            "angle": 0,
            "time": 0
        }
        initial_state = tuple(sorted(initial_state.items()))
        settings = {
            "n_max_sim": 100,
            "solve_depth_init": 10,
            "c": 5,
            "action_space_size": 5, 
            "gamma": 0.995
        }
        more_actions = True
        state = initial_state
        while more_actions:
            for n in range(settings["n_max_sim"]):
                self.simulate(settings,state,settings["solve_depth_init"],self.obs_list)
            max = 0
            best_action = None
            for sap in self.NQ.keys():
                if sap[0] == state:
                    value = self.NQ[sap]["q_val"]
                    if value > max:
                        max = value
                        best_action = sap[1]
            if best_action is None:
                break
            best_sap = (state,best_action)
            result_list.append(best_sap)
            state = self.transition_function(state,best_action)
            print(dict(state)["time"])
            more_actions = len(get_action_space(settings,state,self.obs_list)) != 0
        planned_obs_list = []
        for result in result_list:
            result = dict(result[1])
            result["location"] = dict(result["location"])
            planned_obs_list.append(result)
        return planned_obs_list
    
    def do_search_events(self,planner_inputs):
        self.obs_list = planner_inputs["obs_list"]
        events = planner_inputs["events"]
        plan_start = planner_inputs["plan_start"]
        plan_end = planner_inputs["plan_end"]
        sim_settings = planner_inputs["settings"]
        mcts_plan = []
        initial_state = {
            "angle": 0,
            "time": plan_start
        }
        settings = {
            "n_max_sim": 100,
            "solve_depth_init": 10,
            "c": 5,
            "action_space_size": 5, 
            "gamma": 0.995
        }
        more_actions = True
        state = initial_state
        while more_actions:
            for n in range(settings["n_max_sim"]):
                self.simulate(settings,state,settings["solve_depth_init"],self.obs_list)
            max = 0
            best_action = None
            for nq in self.NQ:
                if nq["sap"]["state"] == state:
                    value = nq["q_val"]
                    if value > max:
                        max = value
                        best_action = nq["sap"]["action"]
            if best_action is None:
                break
            best_sap = {
                "state": state,
                "action": best_action
            }
            mcts_plan.append(best_sap["action"])
            state = self.transition_function(state,best_action)
            next_obs = best_sap["action"]
            for event in events:
                if close_enough(next_obs["location"]["lat"],next_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                    if (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"]):
                        updated_reward = { 
                            "reward": event["severity"]*sim_settings["rewards"]["reward"],
                            "location": next_obs["location"],
                            "last_updated": state["time"]
                        }
                        planner_outputs = {
                            "plan": mcts_plan,
                            "end_time": state["time"],
                            "updated_reward": updated_reward
                        }
                        return planner_outputs
            more_actions = len(get_action_space(settings,state,self.obs_list)) != 0
        planner_outputs = {
            "plan": mcts_plan,
            "end_time": plan_end,
            "updated_reward": None
        }
        return planner_outputs
    
    def do_search_events_interval(self,planner_inputs):
        self.obs_list = planner_inputs["obs_list"]
        new_obs_list = []
        for obs in self.obs_list:
            new_obs = {
                "start": obs["start"],
                "end": obs["end"],
                "angle": obs["angle"],
                "reward": obs["reward"],
                "location": tuple(sorted(obs["location"].items()))
            }
            new_obs_list.append(new_obs)
        self.obs_list = new_obs_list
        events = planner_inputs["events"]
        plan_start = planner_inputs["plan_start"]
        plan_end = planner_inputs["plan_end"]
        self.sim_settings = planner_inputs["settings"]
        mcts_plan = []
        initial_state = {
            "angle": 0,
            "time": plan_start
        }
        initial_state = tuple(sorted(initial_state.items()))
        settings = {
            "n_max_sim": 100,
            "solve_depth_init": 10,
            "c": 5,
            "action_space_size": 5, 
            "gamma": 0.995
        }
        more_actions = True
        state = initial_state
        updated_rewards = []
        while more_actions:
            for n in range(settings["n_max_sim"]):
                self.simulate(settings,state,settings["solve_depth_init"],self.obs_list)
            max = 0
            best_action = None
            for sap in self.NQ.keys():
                if sap[0] == state:
                    value = self.NQ[sap]["q_val"]
                    if value > max:
                        max = value
                        best_action = sap[1]
            if best_action is None:
                break
            best_sap = (state,best_action)
            result = dict(best_sap[1])
            result["location"] = dict(result["location"])
            mcts_plan.append(result)
            state = self.transition_function(state,best_action)
            next_obs = result
            for event in events:
                if close_enough(next_obs["location"]["lat"],next_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                    if (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"]):
                        updated_reward = { 
                            "reward": event["severity"]*settings["rewards"]["reward"],
                            "location": next_obs["location"],
                            "last_updated": dict(state)["time"]
                        }
                        updated_rewards.append(updated_reward)
            more_actions = len(get_action_space(settings,state,self.obs_list)) != 0
            if dict(state)["time"] > plan_end:
                break
        planner_outputs = {
            "plan": mcts_plan,
            "end_time": plan_end,
            "updated_rewards": updated_rewards
        }
        return planner_outputs

    def transition_function(self,state,action):
        action = dict(action)
        new_state = {
            "angle": round(int(action["angle"])/5) * 5,
            "time": int(action["soonest"])
        }
        new_state = tuple(sorted(new_state.items()))
        return new_state
    
    def rollout(self,settings,state,action_space,obs_list,d):
        state = dict(state)
        if d == 0:
            return 0
        if len(action_space) == 0:
            return 0
        else:
            selected_action = action_space[np.random.randint(len(action_space))]
            selected_action = dict(selected_action)
            reward = selected_action["reward"]
            new_state = self.transition_function(state,selected_action)
            return (reward + self.rollout(settings,new_state,get_action_space(settings,new_state,obs_list),obs_list,(d-1)) * np.power(settings["gamma"],selected_action["start"]-state["time"]))

    def simulate(self,settings,state,d,obs_list):
        if d == 0:
            return 0
        state_in_v = False
        for v in self.V:
            if state == v[0]:
                state_in_v = True
        if not state_in_v:
            action_space = get_action_space(settings,state,obs_list)
            if action_space is None:
                return 0
            for action in action_space:
                state_action_pair = (state, tuple(sorted(action.items())))
                self.NQ[state_action_pair] = {
                    "n_val": 1.0,
                    "q_val": 0.0
                }
                self.V.append(state_action_pair)
            return self.rollout(settings,state,action_space,obs_list,settings["solve_depth_init"])
        max = 0.0
        best_action = None
        n_sum = 0
        for action in get_action_space(settings,state,obs_list):
            sap = (state, tuple(sorted(action.items())))
            n_sum += self.NQ[sap]["n_val"]
        for sap in self.NQ.keys():
            if sap[0] == state:
                q_val = self.NQ[sap]["q_val"] + settings["c"]*np.sqrt(np.log10(n_sum)/self.NQ[sap]["n_val"])
                if q_val > max:
                    max = q_val
                    best_action = sap[1]
        if best_action is None:
            return 0
        new_state = self.transition_function(state,best_action)
        r = dict(best_action)["reward"]
        q = r + self.simulate(settings,new_state,d-1,obs_list) * np.power(settings["gamma"],dict(best_action)["start"]-dict(state)["time"])
        # best_sap = {
        #     "state": state,
        #     "action": best_action
        # }
        best_sap = (state, best_action)
        self.NQ[best_sap]["n_val"] += 1
        self.NQ[best_sap]["q_val"] += (q-self.NQ[best_sap]["q_val"]/self.NQ[best_sap]["n_val"])

        return q
