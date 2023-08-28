import json
from orbitpy.mission import Mission
from copy import deepcopy

def main(scenario_dir):
    with open('./missions/base_imaging_sat.json', 'r') as openfile:
        base_satellite = json.load(openfile)
    satellites = []
    r = 4; # number of planes
    s = 4; # number of satellites per plane
    re = 6378
    altitude = 500
    ecc = 0.01
    inc = 67
    argper = 0.0
    for m in range(r):
        for n in range(s):
            new_satellite = {}
            new_satellite = base_satellite.copy()
            new_satellite["@id"] = "imaging_sat_"+str(m)+"_"+str(n)
            new_satellite["name"] = "imaging_sat_"+str(m)+"_"+str(n)
            pu = 360 / (r*s)
            delAnom = pu * r
            delRAAN = pu * s
            RAAN = delRAAN * m
            f = 1
            phasing = pu * f
            
            anom = (n * delAnom + phasing * m)
            new_satellite["orbitState"]["state"]["sma"] = altitude+re
            new_satellite["orbitState"]["state"]["ecc"] = ecc
            new_satellite["orbitState"]["state"]["inc"] = inc
            new_satellite["orbitState"]["state"]["raan"] = RAAN
            new_satellite["orbitState"]["state"]["aop"] = argper
            new_satellite["orbitState"]["state"]["ta"] = anom
            satellites.append(deepcopy(new_satellite))
    with open(scenario_dir +'MissionSpecs.json', 'r') as scenario_specs:
        # load json file as dictionary
        mission_dict = json.load(scenario_specs)
        grid_array = [{"@type": "customGrid", "covGridFilePath": "./coverage_grids/riverATLAS.csv"}]
        mission_dict["grid"] = grid_array
        mission_dict["spacecraft"] = satellites
    out_file = open(scenario_dir+"MissionSpecs.json", "w")
    json.dump(mission_dict,out_file, indent = 4)
    out_file.close()



if __name__ == "__main__":
    main('./missions/test_mission/')