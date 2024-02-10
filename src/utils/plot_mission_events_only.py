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
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


def nearest(items, pivot):
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))

def get_latest_reward_grid_file(settings,step_num):
    timing_list = []
    file_list = []
    for file in os.listdir(settings["directory"]+'reward_grids/'):
        file_list.append(file)
        file = file[:-4]
        file = file [5:]
        timing_list.append(float(file))
    closest_time = None
    time_dist = 86400
    for time in timing_list:
        if step_num >= time and (step_num-time) < time_dist:
            closest_time = time
            time_dist = step_num - time
    idx = timing_list.index(closest_time)
    return file_list[idx]

def plot_step(step_num,settings):
    filename = settings["directory"]+'plots/frame_'+str(step_num).zfill(4)+'.png'
    # m = Basemap(projection='merc',llcrnrlat=-75,urcrnrlat=75,\
    #         llcrnrlon=-180,urcrnrlon=180,resolution='c')
    data_crs = ccrs.PlateCarree()

    # The projection keyword determines how the plot will look
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=data_crs)
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    # ax.set_xlim([-150,-30])
    # ax.set_ylim([20,70])
    x0c, x1c, y0c, y1c = ax.properties()['extent']
    ax.coastlines()

    # ax.yaxis.tick_right()
    # ax.set_xticks([-80,-70,-60,-50,-40], crs=ccrs.PlateCarree())
    # ax.set_yticks([-20,-10,0,10], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # lat_formatter = LatitudeFormatter()
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)

    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
    #                 linewidth=2, color='gray', alpha=0.5, linestyle='--')

    event_rows = []
    with open(settings["directory"]+'events/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            event_rows.append(row)


    for row in event_rows:
        plt.scatter(float(row[1]),float(row[0]),5,marker='^',color='cyan',transform=data_crs)


    

    # legend stuff
    plt.scatter([], [], c='cyan',marker='^', label='Event')


    # Put a legend to the right of the current axis
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', fontsize=12, bbox_to_anchor=(1, 0.5))
    # #m.imshow(precip, origin='upper', cmap='RdYlGn_r', vmin=1, vmax=200, zorder=3)
    
    plt.title('Events at time t='+str(np.round(step_num*settings["time"]["step_size"]/3600,2))+' hours')
    plt.tight_layout()
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
    end_frac = start_frac+settings["plotting"]["plot_duration"]
    num_skip = settings["plotting"]["plot_interval"]
    steps = np.arange(int(np.floor(settings["time"]["duration"]*start_frac*86400/settings["time"]["step_size"])),int(np.floor(settings["time"]["duration"]*end_frac*86400/settings["time"]["step_size"])),num_skip)
    print(steps)
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
    video_name = settings["directory"]+'plots/events.mp4'

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
    end_frac = start_frac+settings["plotting"]["plot_duration"]
    num_skip = settings["plotting"]["plot_interval"]
    steps = np.arange(int(np.floor(settings["time"]["duration"]*start_frac*86400/settings["time"]["step_size"])),int(np.floor(settings["time"]["duration"]*end_frac*86400/settings["time"]["step_size"])),num_skip)
    print(steps)
    missing_steps = []
    for step in steps:
        if not os.path.exists(settings["directory"]+'plots/frame_'+str(step).zfill(4)+'.png'):
            missing_steps.append(step)
    pool = multiprocessing.Pool()
    pool.map(partial(plot_step, settings=settings), missing_steps)

if __name__ == "__main__":
    name = "lightning"
    settings = {
        "name": name,
        "instrument": {
            "ffor": 30,
            "ffov": 0
        },
        "agility": {
            "slew_constraint": "rate",
            "max_slew_rate": 0.1,
            "inertia": 2.66,
            "max_torque": 4e-3
        },
        "orbit": {
            "altitude": 705, # km
            "inclination": 98.4, # deg
            "eccentricity": 0.0001,
            "argper": 0, # deg
        },
        "constellation": {
            "num_sats_per_plane": 2,
            "num_planes": 2,
            "phasing_parameter": 1
        },
        "events": {
            "event_duration": 3600*6,
            "num_events": int(1e5),
            "event_clustering": "clustered"
        },
        "time": {
            "step_size": 10, # seconds
            "duration": 1, # days
            "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
        },
        "rewards": {
            "reward": 10,
            "reward_increment": 0.1,
            "reobserve_reward": 2
        },
        "plotting": {
            "plot_duration": 1,
            "plot_interval": 100
        },
        "planner": "dp",
        "num_meas_types": 3,
        "sharing_horizon": 1000,
        "planning_horizon": 1000,
        "directory": "./missions/"+name+"/",
        "grid_type": "custom", # can be "uniform" or "custom"
        "point_grid": "./coverage_grids/"+name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./src/utils/lightning_events.csv"],
        "process_obs_only": False,
        "conops": "onboard_processing"
    }
    base_directory = settings["directory"]
    timestep = settings["time"]["step_size"]
    duration = settings["time"]["duration"]*86400
    steps = np.arange(0,duration,timestep,dtype=int)
    if not os.path.exists(base_directory+'events'):
        os.mkdir(base_directory+'events')
        if len(settings["event_csvs"]) > 0:
            events = []
            for filename in settings["event_csvs"]:
                with open(filename,newline='') as csv_file:
                    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    i = 0
                    for row in csvreader:
                        if i < 1:
                            i=i+1
                            continue
                        row = [float(i) for i in row[0:5]]
                        events.append(row)
            for i in range(len(steps)):            
                events_per_step = []
                print(i)
                step_time = i*settings["time"]["step_size"] 
                for event in events:
                    if event[2] <= step_time and step_time <= (event[2]+event[3]):
                        event_per_step = [event[0],event[1],event[4]] # lat, lon, start, duration, severity
                        events_per_step.append(event_per_step)
                with open(base_directory+'events/step'+str(i)+'.csv','w') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for event in events_per_step:
                        csvwriter.writerow(event)
    plot_mission(settings)