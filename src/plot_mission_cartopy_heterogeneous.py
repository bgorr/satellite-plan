import time, calendar, datetime
import cv2
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
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

def nearest(items, pivot):
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))

def plot_step(step_num,b):
    filename = b["directory"]+'plots/frame_'+str(step_num).zfill(4)+'.png'
    # m = Basemap(projection='merc',llcrnrlat=-75,urcrnrlat=75,\
    #         llcrnrlon=-180,urcrnrlon=180,resolution='c')
    data_crs = ccrs.PlateCarree()

    # The projection keyword determines how the plot will look
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=data_crs)
    ax.set_global()
    ax.set_extent([-80, -40, -20, 10], crs=ccrs.PlateCarree())
    #ax.set_xlim([-150,-30])
    #ax.set_ylim([20,70])
    x0c, x1c, y0c, y1c = ax.properties()['extent']
    ax.coastlines()
    
    fname = './grwl_files/GRWL_summaryStats.shp'
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(), edgecolor='lightskyblue', facecolor='None', linewidth=0.5)
    ax.add_feature(shape_feature, edgecolor='lightskyblue', facecolor='None', linewidth=0.5)

    ax.yaxis.tick_right()
    ax.set_xticks([-80,-70,-60,-50,-40], crs=ccrs.PlateCarree())
    ax.set_yticks([-20,-10,0,10], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')

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
    # crosslinks = []
    # with open(b["directory"]+'crosslinks/step'+str(step_num)+'.csv','r') as csvfile:
    #     csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     for row in csvreader:
    #         crosslinks.append(row)

    past_lats = []
    past_lons = []
    past_rows = []
    with open(b["directory"]+'constellation_past_observations/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            past_lats.append(float(row[1]))
            past_lons.append(float(row[2]))
            past_rows.append(row)
    grid_lats = []
    grid_lons = []
    if not "point_grid" in b:
        b["point_grid"] = b["directory"]+"orbit_data/grid0.csv"
    with open(b["point_grid"]) as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_lats.append(float(row[0]))
            grid_lons.append(float(row[1]))
    if b["grid_type"] == "event":
        event_rows = []
        with open(b["directory"]+'events/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                event_rows.append(row)
    
    coobs_rows = []
    with open(b["directory"]+'coobs/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            coobs_rows.append(row)


    #ax.drawmapboundary(fill_color='#99ffff')
    #ax.fillcontinents(color='#cc9966',lake_color='#99ffff')
    #ax.stock_img()
    #ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.COASTLINE)

    time = b["initial_datetime"]
    time += datetime.timedelta(seconds=float(b["step_size"]*step_num))
    #ax.add_feature(Nightshade(time, alpha=0.1))

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


    #x, y = ax(grid_lons,grid_lats)meas_types
    ax.scatter(grid_lons,grid_lats,0.5,marker='o',color='blue',transform=data_crs)

    for row in event_rows:
        plt.scatter(float(row[1]),float(row[0]),2,marker='^',color='cyan',transform=data_crs)
        # if float(row[3]) == 0:
        #     plt.scatter(float(row[1]),float(row[0]),int(np.min([4*float(row[2]),10])),marker='*',color='green',transform=data_crs)
        # if float(row[3]) == 1:
        #     plt.scatter(float(row[1]),float(row[0]),int(np.min([4*float(row[2]),10])),marker='*',color='magenta',transform=data_crs)
        # if float(row[3]) == 2:
        #     plt.scatter(float(row[1]),float(row[0]),int(np.min([4*float(row[2]),10])),marker='*',color='cyan',transform=data_crs)
         
    for row in coobs_rows:
        plt.scatter(float(row[2]),float(row[1]),10,marker='s',color='magenta',transform=data_crs)
        # if float(row[3]) == 0:
        #     plt.scatter(float(row[2]),float(row[1]),10,marker='s',color='green',transform=data_crs)
        # if float(row[3]) == 1:
        #     plt.scatter(float(row[2]),float(row[1]),10,marker='s',color='magenta',transform=data_crs)
        # if float(row[3]) == 2:
        #     plt.scatter(float(row[2]),float(row[1]),10,marker='s',color='cyan',transform=data_crs)
         
    #x, y = m(past_lons,past_lats)
    if b["plot_obs"]:
        ax.scatter(past_lons,past_lats,1,marker='o',color='yellow',transform=data_crs)
    satellite_name_dict = {}
    for i in range(b["num_sats_per_plane"]*b["num_planes"]):
        if i < b["num_sats_per_plane"]*b["num_planes"]/b["experiment_settings"]["num_meas_types"]:
            meas_type = 0
        elif i < 2*b["num_sats_per_plane"]*b["num_planes"]/b["experiment_settings"]["num_meas_types"]:
            meas_type = 1
        elif i < 3*b["num_sats_per_plane"]*b["num_planes"]/b["experiment_settings"]["num_meas_types"]:
            meas_type = 2
        elif i < 4*b["num_sats_per_plane"]*b["num_planes"]/b["experiment_settings"]["num_meas_types"]:
            meas_type = 3
        satellite_name_dict["sat"+str(i)] = meas_type
    for row in pos_rows:
        #x, y = ax(float(row[2]),float(row[1]))
        meas_type = satellite_name_dict[row[0]]
        if meas_type == 0:
            color = 'green'
            name = 'img'+row[0]
        elif meas_type == 1:
            color = 'blue'
            name = 'sar'+row[0]
        elif meas_type == 2:
            color = 'red'
            name = 'thermal'+row[0]
        plt.scatter(float(row[2]),float(row[1]),4,marker='^',color=color,transform=data_crs)
        transform = data_crs._as_mpl_transform(ax)
        ax.annotate(name, xy=(float(row[2]), float(row[1])), color=color, xycoords=transform,
                    ha='right', va='top',annotation_clip=True)
        #ax.annotate(row[0], (x, y))
    
    for row in vis_rows:
        #x, y = m(float(row[4]),float(row[3]))
        plt.scatter(float(row[4]),float(row[3]),4,marker='o',color='orange',transform=data_crs)

    if b["plot_obs"]:
        for row in obs_rows:
            #obs_x, obs_y = m(float(row[4]),float(row[3]))
            #m.scatter(obs_x,obs_y,5,marker='o',color='green')
            #sat_x, sat_y = m(float(row[2]),float(row[1]))
            if(np.sign(float(row[4])) != np.sign(float(row[2]))):
                continue
            xs = [float(row[4]),float(row[2])]
            ys = [float(row[3]),float(row[1])]
            plt.plot(xs,ys,linewidth=1,color='r',transform=data_crs)

    if b["plot_obs"]:
        for row in past_rows:
            if int(row[0]) > 1:
                #x, y = m(float(row[2]),float(row[1]))
                transform = data_crs._as_mpl_transform(ax)
                ax.annotate(row[0], xy=(float(row[2]), float(row[1])), xycoords=transform,
                            ha='right', va='top',fontsize=3,annotation_clip=True)

    # for row in crosslinks:
    #     #sat1_x, sat1_y = m(float(row[3]),float(row[2]))
    #     #sat2_x, sat2_y = m(float(row[5]),float(row[4]))
    #     if(np.sign(float(row[5])) != np.sign(float(row[3]))):
    #         continue
    #     xs = [float(row[3]),float(row[5])]
    #     ys = [float(row[2]),float(row[4])]
    #     plt.plot(xs,ys,linewidth=0.5,linestyle='dashed',color='black',transform=data_crs)
        
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
    plt.scatter([], [], c='blue',marker='o', label='Lake location')
    plt.scatter([], [], c='orange',marker='o', label='Point in view')
    plt.scatter([], [], c='green',marker='^', label='Imaging satellite')
    plt.scatter([], [], c='blue',marker='^', label='SAR satellite')
    plt.scatter([], [], c='red',marker='^', label='Thermal satellite')
    plt.scatter([], [], c='magenta',marker='s', label='Co-observation')
    # plt.scatter([], [], c='green',marker='*', label='Lake bloom event')
    # plt.scatter([], [], c='magenta',marker='*', label='Lake temperature event')
    # plt.scatter([], [], c='cyan',marker='*', label='Lake level event')
    # plt.scatter([], [], c='green',marker='s', label='Lake bloom co-obs')
    # plt.scatter([], [], c='magenta',marker='s', label='Lake temperature co-obs')
    # plt.scatter([], [], c='cyan',marker='s', label='Lake level co-obs')
    # plt.plot([],[], c='black', linestyle='dashed', label='Crosslink')


    # Put a legend to the right of the current axis
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))
    # plt.legend(fontsize=5,loc='upper right')
    # #m.imshow(precip, origin='upper', cmap='RdYlGn_r', vmin=1, vmax=200, zorder=3)
    if b["plot_clouds"]:
        ax.imshow(clouds,transform = img_proj, origin='upper', cmap='gray',alpha=0.5)
    if b["plot_rain"]:
        img_extent = (-179.95,179.95, -89.95,89.95)
        ax.imshow(precip, origin='upper', extent=img_extent, cmap=cmap, vmin=0.1*25.4, vmax=752, transform=ccrs.PlateCarree())
    
    plt.title('Simulation state at time t='+str(np.round(step_num*b["step_size"]/3600,2))+' hours')
    plt.tight_layout()
    plt.savefig(filename,dpi=200)
    plt.close()
    print("Step "+str(step_num)+" complete!")

def plot_mission(settings):
    if not os.path.exists(settings["directory"]+'plots/'):
        os.mkdir(settings["directory"]+'plots/')

    # PLOTS THE LAST 1/4th OF THE SIMULATION
    # imageio gif creation kills itself if there are too many images, is there a fix or is it just a WSL issue?
    start_frac = 0
    end_frac = settings["plot_duration"]
    num_skip = settings["plot_interval"]
    steps = np.arange(int(np.floor(settings["duration"]*start_frac*86400/settings["step_size"])),int(np.floor(settings["duration"]*end_frac*86400/settings["step_size"])),num_skip)
    print(steps)
    pool = multiprocessing.Pool()
    #pool.map(partial(plot_step, b=settings), steps)
    plot_missing(settings)
    filenames = []
    for step in steps:
        filenames.append(settings["directory"]+'plots/frame_'+str(step).zfill(4)+'.png')
    print('Charts saved\n')
    # gif_name = settings["directory"]+'animation'
    # # Build GIF
    # print('Creating gif\n')
    # with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    # print('Gif saved\n')

    # os.system("ffmpeg -f image2 -r 1/5 -i ./images/swissGenevaLake%01d.jpg -vcodec mpeg4 -y ./videos/swissGenevaLake.mp4")

    image_folder = settings["directory"]+"plots/"
    video_name = settings["directory"]+'plots/animation.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 5, (width,height))

    for filename in filenames:
        video.write(cv2.imread(filename))

    cv2.destroyAllWindows()
    video.release()

def plot_missing(settings):
    if not os.path.exists(settings["directory"]+'/'):
        os.mkdir(settings["directory"]+'/')
    # PLOTS THE LAST 1/4th OF THE SIMULATION
    # imageio gif creation kills itself if there are too many images, is there a fix or is it just a WSL issue?
    start_frac = 0
    end_frac = settings["plot_duration"]
    num_skip = settings["plot_interval"]
    steps = np.arange(int(np.floor(settings["duration"]*start_frac*86400/settings["step_size"])),int(np.floor(settings["duration"]*end_frac*86400/settings["step_size"])),num_skip)
    print(steps)
    missing_steps = []
    for step in steps:
        if not os.path.exists(settings["directory"]+'plots/frame_'+str(step).zfill(4)+'.png'):
            missing_steps.append(step)
    pool = multiprocessing.Pool()
    pool.map(partial(plot_step, b=settings), missing_steps)

if __name__ == "__main__":
    set_start_method("spawn")
    cross_track_ffor = 60 # deg
    along_track_ffor = 60 # deg
    cross_track_ffov = 0 # deg
    along_track_ffov = 0 # deg
    agility = 1 # deg/s
    num_planes = 6
    num_sats_per_plane = 6
    settings = {
        "directory": "./missions/agu_rain_het/",
        "step_size": 10,
        "duration": 1,
        "plot_interval": 10,
        "plot_duration": 0.1,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "preplanned_observations": None,
        "event_csvs": ["./rain_events_het.csv"],
        "point_grid": "./coverage_grids/agu_rain/event_locations.csv",
        "plot_clouds": False,
        "plot_rain": True,
        "plot_obs": True,
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "planner": "dp",
        "process_obs_only": False,
        "reward": 10,
        "reward_increment": 0.1,
        "reobserve_reward": 2,
        "sharing_horizon": 1000,
        "planning_horizon": 1000,
        "experiment_settings": {
            "ffor": 60,
            "ffov": 0,
            "constellation_size": 6,
            "agility": 1,
            "event_duration": 4*3600,
            "event_frequency": 0.1/3600,
            "event_density": 10,
            "event_clustering": 16,
            "planner": "dp",
            "reobserve_reward": 2,
            "num_meas_types": 3,
            "reward": 10,
            "reward_increment": 0.1,
            "time_horizon": 1000
        }
    }
    plot_mission(settings)