import json
from orbitpy.mission import Mission
from copy import deepcopy
import datetime

def create_mission(settings):
    print("Creating mission")
    with open('./missions/base_imaging_sat.json', 'r') as openfile:
        base_satellite = json.load(openfile)
    with open('./missions/base_sar_sat.json', 'r') as openfile:
        sar_satellite = json.load(openfile)
    with open('./missions/base_thermal_sat.json', 'r') as openfile:
        thermal_satellite = json.load(openfile)
    satellites = []
    new_imaging_instrument = base_satellite["instrument"]
    new_sar_instrument = sar_satellite["instrument"]
    new_thermal_instrument = thermal_satellite["instrument"]

    ### SETTINGS ### TODO: move to a config json?

    new_imaging_instrument["fieldOfViewGeometry"]["angleHeight"] = 60
    new_imaging_instrument["fieldOfViewGeometry"]["angleWidth"] = 60
    new_sar_instrument["fieldOfViewGeometry"]["angleHeight"] = 60
    new_sar_instrument["fieldOfViewGeometry"]["angleWidth"] = 60
    new_thermal_instrument["fieldOfViewGeometry"]["angleHeight"] = 60
    new_thermal_instrument["fieldOfViewGeometry"]["angleWidth"] = 60
    r = 3; # number of planes
    s = 3; # number of satellites per plane
    altitude = 700
    ecc = 0.01
    inc = 67
    argper = 0.0
    f = 1
    #initial_datetime = datetime.datetime(2020,1,1,0,0,0)
    #duration = 0.1 # days
    #step_size = 10 # seconds
    initial_datetime = settings["initial_datetime"]
    step_size = settings["step_size"]
    duration = settings["duration"]
    directory = settings["directory"]

    ### END SETTINGS ###

    re = 6378
    for m in range(r):
        for n in range(s):
            new_satellite = {}
            new_satellite = base_satellite.copy()
            new_satellite["@id"] = "imaging_sat_"+str(m)+"_"+str(n)
            new_satellite["name"] = "imaging_sat_"+str(m)+"_"+str(n)
            pu = 360 / (r*s)
            delAnom = pu * r
            delRAAN = pu * s
            raan = delRAAN * m
            phasing = pu * f
            anom = (n * delAnom + phasing * m)
            new_satellite["orbitState"]["state"]["sma"] = altitude+re
            new_satellite["orbitState"]["state"]["ecc"] = ecc
            new_satellite["orbitState"]["state"]["inc"] = inc
            new_satellite["orbitState"]["state"]["raan"] = raan
            new_satellite["orbitState"]["state"]["aop"] = argper
            new_satellite["orbitState"]["state"]["ta"] = anom
            new_satellite["orbitState"]["date"] = {
                "@type": "GREGORIAN_UT1",
                "year": initial_datetime.year,
                "month": initial_datetime.month,
                "day": initial_datetime.day,
                "hour": initial_datetime.hour,
                "minute": initial_datetime.minute,
                "second": initial_datetime.second
            }
            if m == 0:
                new_satellite["instrument"] = new_imaging_instrument
                new_satellite["name"] = "img_"+str(n)
            if m == 1:
                new_satellite["instrument"] = new_sar_instrument
                new_satellite["name"] = "sar_"+str(n)
            if m == 2:
                new_satellite["instrument"] = new_thermal_instrument
                new_satellite["name"] = "thm_"+str(n)
            satellites.append(deepcopy(new_satellite))
    with open('./missions/base_mission.json', 'r') as scenario_specs:
        # load json file as dictionary
        mission_dict = json.load(scenario_specs)
        if settings["grid_type"] == "static":
            grid_array = [{"@type": "customGrid", "covGridFilePath": "./coverage_grids/riverATLAS.csv"}]
        elif settings["grid_type"] == "event":
            grid_array = [{"@type": "customGrid", "covGridFilePath": "./events/lakes/lake_event_points.csv"}]
        else:
            print("Invalid grid type")
        mission_dict["grid"] = grid_array
        mission_dict["spacecraft"] = satellites
        mission_dict["scenario"]["duration"] = duration
        mission_dict["duration"] = duration
        mission_dict["epoch"] = {
            "@type": "GREGORIAN_UT1",
            "year": initial_datetime.year,
            "month": initial_datetime.month,
            "day": initial_datetime.day,
            "hour": initial_datetime.hour,
            "minute": initial_datetime.minute,
            "second": initial_datetime.second
        }
        mission_dict["propagator"] = {
            "@type": "J2 ANALYTICAL PROPAGATOR",
            "stepSize": step_size
        }
    out_file = open(directory+"MissionSpecs.json", "w")
    json.dump(mission_dict,out_file, indent = 4)
    out_file.close()
    print("Mission created")

if __name__ == "__main__":
    settings = {
        "directory": "./missions/test_mission_5/",
        "step_size": 10,
        "duration": 1,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "event_csvs": ['bloom_events.csv','level_events.csv','temperature_events.csv'],
        "plot_clouds": False,
        "plot_rain": False
    }
    create_mission(settings)