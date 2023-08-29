import datetime
import os

from create_mission import create_mission
from execute_mission import execute_mission
from process_mission import process_mission
from plan_mission import plan_mission
from plot_mission import plot_mission

def main():
    settings = {
        "directory": "./missions/test_mission_3/",
        "step_size": 20,
        "duration": 0.5,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
    }
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
    create_mission(settings)
    execute_mission(settings)
    plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    process_mission(settings)
    plot_mission(settings)


if __name__ == "__main__":
    main()