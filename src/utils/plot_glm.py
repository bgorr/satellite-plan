import netCDF4
import matplotlib.pyplot as plt
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

filepath = "/home/ben/data/noaa-goes17/GLM-L2-LCFA/2022/033/00/OR_GLM-L2-LCFA_G17_s20220330000000_e20220330000200_c20220330000225.nc"

nc = netCDF4.Dataset(filepath)

# examine the variables
print(nc.variables.keys())
#print(nc.variables['event_lat'])

# sample every 10th point of the 'z' variable
lats = nc.variables['event_lat'][:]
lons = nc.variables['event_lon'][:]

def nearest(items, pivot):
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))

data_crs = ccrs.PlateCarree()

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=data_crs)
ax.set_global()
#ax.set_extent([-150, -30, 20, 70], crs=ccrs.PlateCarree())
# ax.set_xlim([-150,-30])
# ax.set_ylim([20,70])
x0c, x1c, y0c, y1c = ax.properties()['extent']
ax.coastlines()

plt.scatter(lons,lats,2,marker='o',color='blue',transform=data_crs)

# Put a legend to the right of the current axis
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc='center left', fontsize=5, bbox_to_anchor=(1, 0.5))
# #m.imshow(precip, origin='upper', cmap='RdYlGn_r', vmin=1, vmax=200, zorder=3)

plt.savefig('./lightning.png',dpi=200)
plt.close()