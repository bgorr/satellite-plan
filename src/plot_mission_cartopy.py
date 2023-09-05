import time, calendar, datetime
from functools import partial
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import urllib, os
import csv
import numpy as np
import imageio
import sys
import multiprocessing
import h5py
import matplotlib.colors as colors
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from multiprocessing import set_start_method


def nearest(items, pivot):
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))

def plot_step(step_num,b):
    filename = f'{b["plot_location"]}/frame_{step_num}.png'
    # m = Basemap(projection='merc',llcrnrlat=-75,urcrnrlat=75,\
    #         llcrnrlon=-180,urcrnrlon=180,resolution='c')
    data_crs = ccrs.PlateCarree()

    # The projection keyword determines how the plot will look
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=data_crs)
    ax.set_global()
    #ax.set_extent([-150, -30, 20, 70], crs=ccrs.PlateCarree())
    # ax.set_xlim([-150,-30])
    # ax.set_ylim([20,70])
    x0c, x1c, y0c, y1c = ax.properties()['extent']
    ax.coastlines()
    #ax.stock_img()
    pos_rows = []
    with open(b["directory"]+'sat_positions/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            pos_rows.append(row)

    vis_rows = []
    with open(b["directory"]+'sat_visibilities/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            vis_rows.append(row)

    obs_rows = []
    with open(b["directory"]+'sat_observations/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            obs_rows.append(row)

    swath_rows = []
    with open(b["directory"]+'ground_swaths/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            swath_rows.append(row)
    crosslinks = []
    with open(b["directory"]+'crosslinks/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            crosslinks.append(row)

    past_lats = []
    past_lons = []
    past_rows = []
    with open(b["directory"]+'constellation_past_observations/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            past_lats.append(float(row[1]))
            past_lons.append(float(row[2]))
            past_rows.append(row)
    if b["grid_type"] == "static":
        grid_lats = []
        grid_lons = []
        with open('./coverage_grids/xgrants_points.csv','r') as csvfile:
            csvreader = csv.reader(csvfile,delimiter=',')
            next(csvfile)
            for row in csvreader:
                grid_lats.append(float(row[0]))
                grid_lons.append(float(row[1]))
    elif b["grid_type"] == "event":
        grid_lats = []
        grid_lons = []
        with open('./events/lakes/lake_event_points.csv','r') as csvfile:
            csvreader = csv.reader(csvfile,delimiter=',')
            next(csvfile)
            for row in csvreader:
                grid_lats.append(float(row[0]))
                grid_lons.append(float(row[1]))

        event_rows = []
        with open(b["directory"]+'events/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                event_rows.append(row)


    #ax.drawmapboundary(fill_color='#99ffff')
    #ax.fillcontinents(color='#cc9966',lake_color='#99ffff')
    #ax.stock_img()
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.COASTLINE)

    time = b["initial_datetime"]
    time += datetime.timedelta(seconds=float(b["step_size"]*step_num))
    ax.add_feature(Nightshade(time, alpha=0.2))

    #ax = plt.gca()

    ### CLOUD SECTION ###
    if b["plot_clouds"]:
        cloud_dir = './clouds/goes16/2020/01/01/ABI-L2-ACMF/'
        cloud_files = []
        cloud_datetimes = []
        for subdir in os.listdir(cloud_dir):
            for f in os.listdir(cloud_dir+subdir):
                starttime_str = f[23:37]
                year = int(starttime_str[0:4])
                days = int(starttime_str[4:7])
                get_month_day = b["initial_datetime"] + datetime.timedelta(days=days-1)
                hour = int(starttime_str[7:9])
                minute = int(starttime_str[9:11])
                seconds = int(starttime_str[11:13])
                file_datetime = datetime.datetime(year,get_month_day.month,get_month_day.day,hour,minute,seconds)
                cloud_files.append(cloud_dir+subdir+"/"+f)
                cloud_datetimes.append(file_datetime)
        if time < cloud_datetimes[0]:
            nearest_cloud_datetime = cloud_datetimes[0]
        else:
            nearest_cloud_datetime = nearest(cloud_datetimes,time)
        idx = cloud_datetimes.index(nearest_cloud_datetime)
        nearest_filename = cloud_files[idx]
        print(nearest_filename)
        nc = Dataset(nearest_filename) #filename (the ".h5" file)  

        clouds = nc.variables["BCM"]
        proj_var = nc.variables['goes_imager_projection']
        initial_datetime = datetime.datetime(2000,1,1,12,0,0)
        times = nc.variables["time_bounds"][:]
        actual_datetime = initial_datetime + datetime.timedelta(seconds=times[0])
        sat_height = proj_var.perspective_point_height
        central_lon = proj_var.longitude_of_projection_origin
        semi_major = proj_var.semi_major_axis
        semi_minor = proj_var.semi_minor_axis
        semi_minor = proj_var.semi_minor_axis

        clouds = np.ma.masked_less(clouds, 0.5)
        globe = ccrs.Globe(semimajor_axis=semi_major, semiminor_axis=semi_minor)
        img_proj = ccrs.Geostationary(central_longitude = central_lon, satellite_height=sat_height,globe=globe)

    ### END CLOUD SECTION ### 

    ### RAIN SECTION ###
    if b["plot_rain"]:
        a = np.array([0,.01,.1,.25,.5,1,1.5,2,3,4,6,8,10,15,20,30])/25.4

        # Normalize the bin between 0 and 1 (uneven bins are important here)
        norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]

        # Color tuple for every bin
        C = np.array([[255,255,255],
                    [199,233,192],
                    [161,217,155],
                    [116,196,118],
                    [49,163,83],
                    [0,109,44],
                    [255,250,138],
                    [255,204,79],
                    [254,141,60],
                    [252,78,42],
                    [214,26,28],
                    [173,0,38],
                    [112,0,38],
                    [59,0,48],
                    [76,0,115],
                    [255,219,255]])

        # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
        COLORS = []
        for i, n in enumerate(norm):
            COLORS.append((n, np.array(C[i])/255.))

        # Create the colormap
        cmap = colors.LinearSegmentedColormap.from_list("precipitation", COLORS)

        time = b["initial_datetime"] + datetime.timedelta(seconds=float(10*step_num))
        base_filename = "3B-HHR-L.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V06B.HDF5"
        seconds_elapsed = b["step_size"]*step_num
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
            precip = f['/Grid/precipitationCal'][:]
            precip = np.flip( precip[0,:,:].transpose(), axis=0 )
            # get _FillValue for data masking
            #img_arr_fill = f[image].attrs['_FillValue'][0]   
            precip = np.ma.masked_less(precip, 0.1*25.4)
            precip = np.ma.masked_greater(precip, 752)
    ### END RAIN SECTION ###


    #x, y = ax(grid_lons,grid_lats)
    ax.scatter(grid_lons,grid_lats,2,marker='o',color='blue',transform=data_crs)
    if b["grid_type"] == "event":
        for row in event_rows:
            plt.scatter(float(row[1]),float(row[0]),int(np.min([4*float(row[2]),10])),marker='*',color='cyan',transform=data_crs)

    #x, y = m(past_lons,past_lats)
    ax.scatter(past_lons,past_lats,3,marker='o',color='yellow',transform=data_crs)
    
    for row in pos_rows:
        #x, y = ax(float(row[2]),float(row[1]))
        plt.scatter(float(row[2]),float(row[1]),4,marker='^',color='black',transform=data_crs)
        transform = data_crs._as_mpl_transform(ax)
        ax.annotate(row[0], xy=(float(row[2]), float(row[1])), xycoords=transform,
                    ha='right', va='top',annotation_clip=True)
        #ax.annotate(row[0], (x, y))
    
    for row in vis_rows:
        #x, y = m(float(row[4]),float(row[3]))
        plt.scatter(float(row[4]),float(row[3]),4,marker='o',color='orange',transform=data_crs)

    for row in obs_rows:
        #obs_x, obs_y = m(float(row[4]),float(row[3]))
        #m.scatter(obs_x,obs_y,5,marker='o',color='green')
        #sat_x, sat_y = m(float(row[2]),float(row[1]))
        if(np.sign(float(row[4])) != np.sign(float(row[2]))):
            continue
        xs = [float(row[4]),float(row[2])]
        ys = [float(row[3]),float(row[1])]
        plt.plot(xs,ys,linewidth=1,color='r',transform=data_crs)

    for row in past_rows:
        if int(row[0]) > 1:
            #x, y = m(float(row[2]),float(row[1]))
            transform = data_crs._as_mpl_transform(ax)
            ax.annotate(row[0], xy=(float(row[2]), float(row[1])), xycoords=transform,
                        ha='right', va='top',fontsize=5,annotation_clip=True)

    for row in crosslinks:
        #sat1_x, sat1_y = m(float(row[3]),float(row[2]))
        #sat2_x, sat2_y = m(float(row[5]),float(row[4]))
        if(np.sign(float(row[5])) != np.sign(float(row[3]))):
            continue
        xs = [float(row[3]),float(row[5])]
        ys = [float(row[2]),float(row[4])]
        plt.plot(xs,ys,linewidth=0.5,linestyle='dashed',color='black',transform=data_crs)
        
    satlist = []
    for i in range(len(swath_rows)):
        if swath_rows[i][0] not in satlist:
            satlist.append(swath_rows[i][0])

    for sat in satlist:
        specific_sat = [x for x in swath_rows if sat == x[0]]
        xs = []
        ys = []
        longitude_sign = np.sign(float(specific_sat[0][2]))
        for row in specific_sat:
            #x, y = m(float(row[2]),float(row[1]))
            if (not np.sign(float(row[2])) == longitude_sign) and (np.abs(float(row[2])) > 90.0):
                xs = []
                ys = []
                break
            xs.append(float(row[2]))
            ys.append(float(row[1]))
        #x, y = m(float(specific_sat[0][2]),float(specific_sat[0][1]))
        xs.append(float(specific_sat[0][2]))
        ys.append(float(specific_sat[0][1]))
        plt.plot(xs,ys,linewidth=0.5,color='purple',transform=ccrs.Geodetic())

    

    # legend stuff
    plt.scatter([], [], c='blue',marker='o', label='Grid location')
    plt.scatter([], [], c='orange',marker='o', label='Point in view')
    plt.scatter([], [], c='yellow',marker='o', label='Point observed')
    plt.scatter([], [], c='black',marker='^', label='Satellite')
    plt.plot([],[], c='black', linestyle='dashed', label='Crosslink')
    plt.plot([],[], c='red', label='Observation')


    # Put a legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', fontsize=5, bbox_to_anchor=(1, 0.5))
    # #m.imshow(precip, origin='upper', cmap='RdYlGn_r', vmin=1, vmax=200, zorder=3)
    if b["plot_clouds"]:
        ax.imshow(clouds,transform = img_proj, origin='upper', cmap='gray',alpha=0.5)
    if b["plot_rain"]:
        ax.imshow(precip, origin='upper', extent=[x0c, x1c, y0c, y1c], cmap=cmap, vmin=0.01*25.4, vmax=762)
    
    plt.title('Simulation state at time t='+str(step_num)+' steps')
    plt.savefig(filename,dpi=200)
    plt.close()
    print("Step "+str(step_num)+" complete!")

def plot_mission(settings):
    if not os.path.exists(settings["directory"]+'plots/'):
        os.mkdir(settings["directory"]+'plots/')
    pool = multiprocessing.Pool()
    # PLOTS THE LAST 1/4th OF THE SIMULATION
    # imageio gif creation kills itself if there are too many images, is there a fix or is it just a WSL issue?
    start_frac = 0
    end_frac = settings["plot_duration"]
    num_skip = settings["plot_interval"]
    steps = np.arange(int(np.floor(settings["duration"]*start_frac*86400/settings["step_size"])),int(np.floor(settings["duration"]*end_frac*86400/settings["step_size"])),num_skip)
    print(steps)
    pool.map(partial(plot_step, b=settings), steps)
    filenames = []
    for step in steps:
        filenames.append(f'{settings["directory"]}plots/frame_{step}.png')
    # print('Charts saved\n')
    gif_name = settings["plot_location"]+'animation'
    # Build GIF
    print('Creating gif\n')
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')

if __name__ == "__main__":
    set_start_method("spawn")
    settings = {
        "directory": "./missions/test_mission_6/",
        "step_size": 1,
        "duration": 2/24,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "static", # can be "event" or "static"
        "preplanned_observations": "./missions/test_mission_6/planner_outputs/accesses_2h_rew_5sat_sol_2degs.csv",
        "event_csvs": [],
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": True
    }
    plot_mission(settings)