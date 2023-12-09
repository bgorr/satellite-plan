import csv
import os
import numpy as np
import datetime
import h5py

def get_nearest_cell(lat,lon,grid,lats,lons):
    # print(lat)
    # print(lon)
    lat_min = -89.95
    lat_max = 89.95
    lon_min = -179.95
    lon_max = 179.95
    lat_frac = (lat - lat_min)/(lat_max - lat_min)
    lon_frac = (lon - lon_min)/(lon_max - lon_min)
    # print(lat_frac)
    # print(lon_frac)
    row = int(lat_frac*(len(grid[:,0])-1))
    col = int(lon_frac*(len(grid[0,:])-1))
    # print(row)
    # print(col)
    # print(lats[row])
    # print(lons[col])
    #print(np.shape(grid))
    return grid[row,col]

filename = './src/utils/grwl_river_output.csv'

river_locations = []
with open(filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 5:
            i=i+1
            continue
        row = [float(i) for i in row]
        river_locations.append(row)

rain_steps = len(os.listdir('./rain_data/'))
initial_datetime = datetime.datetime(2020,1,1,0,0,0)
times = np.linspace(0,86400,rain_steps)
events = []
het_events = []
precip_grids = []
lat_grids = []
lon_grids = []
for step_num in range(rain_steps-1):
    seconds_elapsed = times[step_num]
    time = initial_datetime + datetime.timedelta(seconds=seconds_elapsed)
    base_filename = "3B-HHR-L.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V06B.HDF5"
    half_hours_from_midnight = np.floor(seconds_elapsed / 1800)
    half_hours_in_minutes = int(30*half_hours_from_midnight)
    if (half_hours_from_midnight % 2) == 1:
        minutes_str = "30"
        end_minutes_str = "59"
    else:
        minutes_str = "00"
        end_minutes_str = "29"
    hours_str = str(int(np.floor(half_hours_from_midnight / 2)))
    rain_filename = base_filename[0:23]+str(time.year)+str(time.month).zfill(2)+str(time.day).zfill(2)+"-S"+hours_str.zfill(2)+minutes_str+"00-E"+hours_str.zfill(2)+end_minutes_str+"59."+str(half_hours_in_minutes).zfill(4)+base_filename[52:]
    
    fn = './rain_data/'+rain_filename #filename (the ".h5" file)
    with h5py.File(fn) as f:      
        # retrieve image data:
        #print(f['Grid/lat'][:])
        precip = f['/Grid/precipitationCal'][:]
        precip = precip[0,:,:].transpose()
        lat = f['/Grid/lat'][:]
        lon = f['/Grid/lon'][:]
        # get _FillValue for data masking
        #img_arr_fill = f[image].attrs['_FillValue'][0]   
        precip = np.ma.masked_less(precip, 0.1*25.4)
        precip = np.ma.masked_greater(precip, 752)
    precip_grids.append(precip)
    lat_grids.append(lat)
    lon_grids.append(lon)
for loc in river_locations:
    event_occurring = False
    event_start = 0
    for step_num in range(rain_steps-1):
        precip = precip_grids[step_num]
        lat = lat_grids[step_num]
        lon = lon_grids[step_num]
        precip_val = get_nearest_cell(loc[1],loc[0],precip,lat,lon)
        if event_occurring == True:
            if not (precip_val > 0):
                event_occurring = False
                event = [loc[1],loc[0],event_start,times[step_num]-event_start,1]
                meas_types_needed = [0,1,2]
                het_event = [loc[1],loc[0],event_start,times[step_num]-event_start,1,meas_types_needed]
                # event = {
                #     "start": event_start,
                #     "end": times[step_num],
                #     "location": {
                #         "lat": loc[1],
                #         "lon": loc[0]
                #     }
                # }
                events.append(event)
                het_events.append(het_event)
            elif step_num == rain_steps-2:
                event_occurring = False
                event = [loc[1],loc[0],event_start,times[step_num]-event_start,1]
                meas_types_needed = [0,1,2]
                het_event = [loc[1],loc[0],event_start,times[step_num]-event_start,1,meas_types_needed]
                # event = {
                #     "start": event_start,
                #     "end": times[step_num],
                #     "location": {
                #         "lat": loc[1],
                #         "lon": loc[0]
                #     }
                # }
                events.append(event)
                het_events.append(het_event)
        else:
            if precip_val > 0:
                event_occurring = True
                event_start = times[step_num]

with open('./rain_events.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
    for event in events:
        csvwriter.writerow(event)

with open('./rain_events_het.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity','meas_types_needed'])
    for event in het_events:
        csvwriter.writerow(event)
