import datetime
import os
import numpy as np
import csv
import time
import shutil, errno
import random
import tqdm
from scipy.stats import qmc
import sys
sys.path.append(".")


def prep_experiment():
    simulation_duration = 1 # days
    event_duration = 3600*3
    num_event_locations = 1000
    num_events = 100
    possible_event_locations = []
    center_lats = np.random.uniform(-90,90,100)
    center_lons = np.random.uniform(-180,180,100)
    for i in range(len(center_lons)):
        var = 1
        mean = [center_lats[i], center_lons[i]]
        cov = [[var, 0], [0, var]]
        num_points_per_cell = int(6.48e6/100)
        xs, ys = np.random.multivariate_normal(mean, cov, num_points_per_cell).T
        for i in range(len(xs)):
            location = [xs[i],ys[i]]
            possible_event_locations.append(location)
    event_locations = []
    for i in range(num_event_locations):
        event_locations.append(random.choice(possible_event_locations))
    for i in range(5):
        if not os.path.exists("./events/http_events/events"+str(i)+".csv"):
            events = []
            for j in range(num_events):
                event_location = random.choice(event_locations)
                step = int(np.random.uniform(0,simulation_duration*86400))
                event = [event_location[0],event_location[1],step,event_duration,1]
                events.append(event)
            if not os.path.exists("./events/http_events/"):
                os.mkdir("./events/http_events/")
            with open("./events/http_events/events"+str(i)+".csv",'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
                for event in events:
                    csvwriter.writerow(event)
    
    if not os.path.exists("./coverage_grids/http_events/coverage_grids/"):
        os.mkdir("./coverage_grids/http_events/")
        with open("./coverage_grids/http_events/event_locations.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]'])
            for location in event_locations:
                csvwriter.writerow(location)


def main():
    prep_experiment()


if __name__ == '__main__':
	main()