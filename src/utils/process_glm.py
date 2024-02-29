import netCDF4
import os
import csv
import numpy as np

directories = ["/home/ben/data/noaa-goes17/GLM-L2-LCFA/2022/033/","/home/ben/data/noaa-goes16/GLM-L2-LCFA/2022/033/"]

event_locations = {}
for directory in directories:
    for subdir in os.listdir(directory):
        for filepath in os.listdir(directory+subdir):
            file_time_hours = float(filepath[27:29])
            file_time_minutes = float(filepath[29:31])
            file_time_seconds = float(filepath[31:33])
            file_time = file_time_hours*3600+file_time_minutes*60+file_time_seconds
            nc = netCDF4.Dataset(directory+subdir+"/"+filepath)

        lats = nc.variables['group_lat'][:]
        lons = nc.variables['group_lon'][:]
        time_offsets = nc.variables['group_time_offset'][:]
        for i in range(len(lats)):
            updated_time = file_time+time_offsets[i]
            location = (np.round(lats[i],1),np.round(lons[i],1))
            if location not in event_locations:
                event_locations[location] = [updated_time]
            else:
                event_locations[location].append(updated_time)

rows = []
for event_loc in event_locations.keys():
    times = event_locations[event_loc]
    earliest_time = np.min(times)
    latest_time = np.max(times)
    time_arrays = []
    through_all_times = False
    #print(times)
    while not through_all_times:
        initial_times = times
        for i in range(1,len(times)):
            if times[i] - times[i-1] > 3600:
                time_arrays.append(times[0:i-1])
                times = times[i:]
                break
        if times == initial_times:
            time_arrays.append(times)
            through_all_times = True
    #print(time_arrays)
    for array in time_arrays:
        if len(array) > 0:
            earliest_time = np.min(array)
            latest_time = np.max(array)
            rows.append([event_loc[0],event_loc[1],earliest_time,latest_time-earliest_time,1])

with open("./src/utils/lightning_events.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)