import netCDF4
import os
import csv
import numpy as np
import datetime


directory = "./trmm_download/"

# print(nc.variables)
# print(nc.variables['lightning_area_location'][:])
# print(nc.variables['lightning_event_location'][:])
# print(len(nc.variables['lightning_event_location'][:]))
# print(nc.variables['lightning_group_location'][:])
# print(len(nc.variables['lightning_group_location'][:]))
# print(nc.variables['lightning_group_TAI93_time'][:])
# print(nc.variables['lightning_group_observe_time'][:])

tai93start = datetime.datetime(1993,1,1,0,0,0)
day_start = datetime.datetime(2015,2,2,0,0,0)

event_locations = {}

for filepath in os.listdir(directory):
    if ".nc" in filepath:
        nc = netCDF4.Dataset(directory+"/"+filepath)
        lats = nc.variables['lightning_group_lat'][:]
        lons = nc.variables['lightning_group_lon'][:]
        time_diffs = nc.variables['lightning_group_TAI93_time'][:]
        durations = nc.variables['lightning_group_observe_time'][:]
        for i in range(len(lats)):
            lightning_time = tai93start + datetime.timedelta(seconds=time_diffs[i])
            start_time = (lightning_time - day_start).total_seconds()
            location = (np.round(lats[i],1),np.round(lons[i],1))
            if location not in event_locations:
                event_locations[location] = [start_time]
            else:
                event_locations[location].append(start_time)

rows = []
for event_loc in event_locations.keys():
    times = event_locations[event_loc]
    earliest_time = np.min(times)
    latest_time = np.max(times)
    rows.append([event_loc[0],event_loc[1],earliest_time,latest_time-earliest_time,1])

with open("./src/utils/trmm_lightning_events.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)