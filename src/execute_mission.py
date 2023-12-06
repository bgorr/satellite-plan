import json
from orbitpy.mission import Mission
import datetime
import os

def execute_mission(settings):
    print("Executing mission")
    
    scenario_dir = settings["directory"]
    data_dir = settings["directory"] + '/orbit_data/'
    if os.path.exists(data_dir+'comm/'):
        print("Skipping mission execution")
        return
    with open(scenario_dir +'MissionSpecs.json', 'r') as scenario_specs:
        # load json file as dictionary
        mission_dict = json.load(scenario_specs)

        # save specifications of propagation in the orbit data directory
        with open(data_dir +'MissionSpecs.json', 'w') as mission_specs:
            mission_specs.write(json.dumps(mission_dict, indent=4))

        # set output directory to orbit data directory
        if mission_dict.get("settings", None) is not None:
            mission_dict["settings"]["outDir"] = scenario_dir + '/orbit_data/'
        else:
            mission_dict["settings"] = {}
            mission_dict["settings"]["outDir"] = scenario_dir + '/orbit_data/'
        # propagate data and save to orbit data directory
        print("Executing mission in OrbitPy...")
        mission = Mission.from_json(mission_dict)  
        mission.execute()                
        print("Mission executed!")

if __name__ == "__main__":
    settings = {
        "directory": "./missions/test_mission_5_reduced/",
        "step_size": 10,
        "duration": 1,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "event_csvs": ['bloom_events_reduced.csv','level_events_reduced.csv','temperature_events_reduced.csv'],
        "plot_clouds": False,
        "plot_rain": False
    }
    execute_mission(settings)