import datetime
import os

from utils.create_mission import create_mission
from utils.execute_mission import execute_mission
from utils.plan_mission import plan_mission
from satplan.process_mission import process_mission
from satplan.plot_mission import plot_mission

def main(settings : dict):
    
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
    # create_mission(settings)
    # execute_mission(settings)
    # plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    process_mission(settings)
    plot_mission(settings)


if __name__ == "__main__":
    settings = {
        "directory": "./missions/test_mission_3/",
        "step_size": 20,
        "duration": 0.5,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
    }
    main(settings)