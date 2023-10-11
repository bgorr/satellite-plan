import os
import csv
from planners import utils
from tqdm import tqdm
from multiprocessing import Pool
import config

def convert_geo_cords(settings):
    orbit_data_dir = settings["directory"] + 'orbit_data/'
    compute_dirs = []
    for subdir in os.listdir(orbit_data_dir):
        if "comm" in subdir or ".json" in subdir:
            continue
        sat_files = os.listdir(orbit_data_dir + subdir)
        if 'state_geo.csv' in sat_files:
            # os.remove(orbit_data_dir + subdir + "/" + 'state_geo.csv')
            continue
        compute_dirs.append(subdir)

    # Use a Pool to parallelize the computation
    if len(compute_dirs) == 0:
        return
    print('--> CONVERTING TO GEO COORDINATES', config.cores)
    with Pool(processes=config.cores) as pool:
        # We use starmap since our function has multiple arguments
        list(tqdm(pool.starmap(convert_geo_cords_sat, [(orbit_data_dir, subdir) for subdir in compute_dirs]),
                  total=len(compute_dirs)))


def convert_geo_cords_sat(orbit_data_dir, subdir):
    sat_files = os.listdir(orbit_data_dir + subdir)
    for f in sat_files:
        if 'cartesian' in f:
            row_data = []
            with open(orbit_data_dir + subdir + "/" + f, newline='') as csv_file:
                spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for idx, row in enumerate(spamreader):
                    if idx < 5:
                        continue
                    row = [float(i) for i in row]
                    row_data.append(row)
            with open(orbit_data_dir + subdir + "/" + 'state_geo.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['seconds', 'lat', 'lon'])
                for idx, row in enumerate(row_data):
                    time = row[0] * 10  # seconds elapsed since start date
                    x = row[1] * 1000  # km to m
                    y = row[2] * 1000  # km to m
                    z = row[3] * 1000  # km to m
                    lat, lon = utils.eci_to_latlon(x, y, z, time)
                    csvwriter.writerow([time, lat, lon])





