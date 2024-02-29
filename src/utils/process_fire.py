import netCDF4
import os
import csv
import numpy as np
import tqdm

def calculate_degrees(file_id):
    
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables['y'][:]  # N/S elevation angle in radians
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lat, abi_lon
directories = ["/home/ben/data/noaa-goes17/ABI-L2-FDCF/2022/033/","/home/ben/data/noaa-goes16/ABI-L2-FDCF/2022/033/"]
#OR_ABI-L2-FDCF-M6_G16_s20220332010207
event_locations = {}
for directory in directories:
    for subdir in os.listdir(directory):
        for filepath in os.listdir(directory+subdir):
            print(filepath)
            file_time_hours = float(filepath[30:32])
            print(file_time_hours)
            file_time_minutes = float(filepath[32:34])
            file_time_seconds = float(filepath[34:36])
            file_time = file_time_hours*3600+file_time_minutes*60+file_time_seconds
            print(file_time)
            nc = netCDF4.Dataset(directory+subdir+"/"+filepath)
            lats, lons = calculate_degrees(nc)
            fire_flags = nc.variables['Mask'][:][:]
            fire_pixel_indices = np.where((fire_flags == 10) | (fire_flags == 11) | (fire_flags == 30) | (fire_flags == 31))
            for i in range(len(fire_pixel_indices[0])):
                x_coord = fire_pixel_indices[0][i]
                y_coord = fire_pixel_indices[1][i]
                lat = lats[x_coord,y_coord]
                lon = lons[x_coord,y_coord]
                location = (np.round(lat,1),np.round(lon,1))
                if location not in event_locations:
                    event_locations[location] = [file_time]
                else:
                    event_locations[location].append(file_time)
            # for i in range(len(lats)):
            #     print(i)
            #     for j in range(len(lons)):
            #         if (fire_flags[i,j] == 10) or (fire_flags[i,j] == 11) or (fire_flags[i,j] == 30) or (fire_flags[i,j] == 31):
            #             location = (np.round(lats[i],1),np.round(lons[j],1))
            #             if location not in event_locations:
            #                 event_locations[location] = [file_time]
            #             else:
            #                 event_locations[location].append(file_time)

rows = []
for event_loc in event_locations.keys():
    times = event_locations[event_loc]
    earliest_time = np.min(times)
    latest_time = np.max(times)
    duration = latest_time - earliest_time
    if duration == 0:
        duration = 600
    rows.append([event_loc[0],event_loc[1],earliest_time,duration,1])

with open("./src/utils/fire_events.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)