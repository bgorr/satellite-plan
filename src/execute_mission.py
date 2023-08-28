import json
from orbitpy.mission import Mission

def main(scenario_dir):
    data_dir = scenario_dir + 'orbit_data/'
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
    main('./missions/test_mission/')