import numpy as np
import datetime
import csv
import os
import multiprocessing
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import config
from utils.compute_experiment_statistics import compute_statistics

from planners import utils



class ExperimentResult:
    def __init__(self, settings):
        self.settings = settings

    def run(self, epoch=0):
        directory = self.settings["directory"] + "orbit_data/"
        satellites = []
        all_initial_observations = []
        for subdir in os.listdir(directory):
            satellite = {}
            if "comm" in subdir:
                continue
            if ".json" in subdir:
                continue
            for f in os.listdir(directory + subdir):
                if "state_cartesian" in f:
                    with open(directory + subdir + "/" + f, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        states = []
                        i = 0
                        for row in spamreader:
                            if i < 5:
                                i = i + 1
                                continue
                            row = [float(i) for i in row]
                            states.append(row)
                    satellite["orbitpy_id"] = subdir
                    satellite["states"] = states

                if "datametrics" in f:
                    with open(directory + subdir + "/" + f, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        visibilities = []
                        i = 0
                        for row in spamreader:
                            if i < 5:
                                i = i + 1
                                continue
                            row[2] = "0.0"
                            row = [float(i) for i in row]
                            row.append(subdir)
                            visibilities.append(row)
                    satellite["visibilities"] = visibilities
                    # all_visibilities.extend(visibilities)

                if "plan" in f and not "replan" in f and self.settings["planner"] in f and str(epoch) in f:
                    with open(directory + subdir + "/" + f, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        observations = []
                        i = 0
                        for row in spamreader:
                            if i < 1:
                                i = i + 1
                                continue
                            row = [float(i) for i in row]
                            row.append(subdir)
                            observations.append(row)
                    all_initial_observations.extend(observations)

            if self.settings["preplanned_observations"] is not None:
                with open(self.settings["preplanned_observations"], newline='') as csv_file:
                    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    observations = []
                    i = 0
                    for row in csvreader:
                        if i < 1:
                            i = i + 1
                            continue
                        if int(row[0][8:]) == int(satellite["orbitpy_id"][3]):
                            obs = [int(float(row[3])), int(float(row[4])), float(row[1]) * 180 / np.pi,
                                   float(row[2]) * 180 / np.pi]
                            observations.append(obs)
                satellite["observations"] = observations

            if "orbitpy_id" in satellite:
                satellites.append(satellite)


        prior_visibilities = [sat['visibilities'] for sat in satellites]
        all_visibilities = []
        with Pool(processes=config.cores) as pool:
            sat_visibilities = list(
                tqdm(pool.imap(utils.init_sat_observations_post_proc, prior_visibilities), total=len(prior_visibilities)))
        for sat_visibility in sat_visibilities:
            all_visibilities.extend(sat_visibility)


        events = []
        event_filename = self.settings["event_csvs"][0]
        with open(event_filename, newline='') as csv_file:
            csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
            i = 0
            for row in csvreader:
                if i < 1:
                    i = i + 1
                    continue
                events.append(row)  # lat, lon, start, duration, severity

        print("\nPlatform Event Observations")
        init_results = compute_statistics(events, all_initial_observations, self.settings)

        print("\nPotential Event Observations (visibilities)")
        vis_results = compute_statistics(events, all_visibilities, self.settings)

        overall_results = {
            "init_results": init_results,
            "vis_results": vis_results,
            "num_events": len(events),
            "num_obs_init": len(all_initial_observations),
            "num_vis": len(all_visibilities)
        }
        return overall_results

if __name__ == "__main__":
    pass


