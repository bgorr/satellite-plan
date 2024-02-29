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
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator, LongitudeLocator)
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


def plot_step(step_num,b):
    filename = f'{b["directory"]}plots/frame_{step_num}.png'
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

    ax.yaxis.tick_right()
    ax.set_xticks([-80,-70,-60,-50,-40], crs=ccrs.PlateCarree())
    ax.set_yticks([-20,-10,0,10], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')

    ax.coastlines()

    fname = './grwl_files/GRWL_summaryStats.shp'
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(), edgecolor='lightskyblue', facecolor='None', linewidth=0.5)
    ax.add_feature(shape_feature, edgecolor='lightskyblue', facecolor='None', linewidth=0.5)

    time = b["initial_datetime"]
    time += datetime.timedelta(seconds=float(b["step_size"]*step_num))
    #ax.add_feature(Nightshade(time, alpha=0.2))

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

    ax.scatter(grid_lons,grid_lats,0.5,marker='o',color='black',transform=data_crs)

    event_rows = []
    with open(b["directory"]+'events/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            event_rows.append(row)

    for row in event_rows:
        plt.scatter(float(row[1]),float(row[0]),int(np.min([4*float(row[2]),10])),marker='*',color='blue',transform=data_crs)

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

    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width, box.height])
    img_extent = (-179.95,179.95, -89.95,89.95)
    ax.imshow(precip, origin='upper', extent=img_extent, cmap=cmap, vmin=0.1*25.4, vmax=752, transform=ccrs.PlateCarree())
    
    # legend stuff
    plt.scatter([], [], c='blue',marker='o', label='River location')
    plt.scatter([], [], c='black',marker='o',label='Event location')

    # plt.scatter([], [], c='orange',marker='o', label='Point in view')
    # plt.scatter([], [], c='green',marker='^', label='Imaging satellite')
    # plt.scatter([], [], c='blue',marker='^', label='SAR satellite')
    # plt.scatter([], [], c='red',marker='^', label='Thermal satellite')
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
    ax.legend(loc='center left', fontsize=5, bbox_to_anchor=(1, 0.5))

    plt.title('Simulation state at time t='+str(np.round(step_num*b["step_size"]/3600,2))+' hours')
    plt.savefig(filename,dpi=300)
    plt.close()
    print("Step "+str(step_num)+" complete!")

def plot_mission(settings):
    # if not os.path.exists(settings["directory"]):
    #     os.mkdir(settings["directory"])
    # if not os.path.exists(settings["directory"]+'plots/'):
    #     os.mkdir(settings["directory"]+'plots/')
    pool = multiprocessing.Pool()
    # imageio gif creation kills itself if there are too many images, is there a fix or is it just a WSL issue?
    start_frac = 0
    num_skip = 100
    steps = np.arange(int(np.floor(settings["time"]["duration"]*start_frac*86400/settings["time"]["step_size"])),int(np.floor(settings["time"]["duration"]*86400/settings["time"]["step_size"])),num_skip)
    # print(steps)
    #pool.map(partial(plot_step, b=settings), steps)
    plot_missing(settings)
    filenames = []
    for step in steps:
        filenames.append(f'{settings["directory"]}plots/frame_{step}.png')
    # print('Charts saved\n')
    gif_name = settings["directory"]+'animation'
    # Build GIF
    print('Creating gif\n')
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Gif saved\n')

def plot_missing(settings):
    if not os.path.exists(settings["directory"]+'/'):
        os.mkdir(settings["directory"]+'/')
    # PLOTS THE LAST 1/4th OF THE SIMULATION
    # imageio gif creation kills itself if there are too many images, is there a fix or is it just a WSL issue?
    start_frac = 0
    num_skip = 100
    steps = np.arange(int(np.floor(settings["time"]["duration"]*start_frac*86400/settings["time"]["step_size"])),int(np.floor(settings["time"]["duration"]*86400/settings["time"]["step_size"])),num_skip)
    print(steps)
    for step in steps:
        if not os.path.exists(f'{settings["directory"]}plots/frame_{step}.png'):
            plot_step(step,settings)


def process_mission(settings):
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
                    row = [float(i) for i in row]
                    events.append(row)
        for i in range(len(steps)):            
            events_per_step = []
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

if __name__ == "__main__":
    set_start_method("spawn")
    settings = {
        "directory": "./missions/plot_rain/",
        "step_size": 10,
        "duration": 1,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "event_csvs": ["./rain_events.csv"],
        "point_grid": "./coverage_grids/agu_rain/event_locations.csv"
    }
    process_mission(settings)
    plot_mission(settings)