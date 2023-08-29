import numpy as np
import h5py
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# file='./plotting_data/3B-HHR-L.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V06B.HDF5'
# data = h5py.File(file,'r')

# # -- extract the 3600x1800 element precipitation array.
# # For Version 6 IMERG HDF5 files, read the "precipitationCal"
# # variable if it is a half-hour file and the "precipitation"
# # variable if it is a monthly file.  For Version 7, the variable
# # is "precipitation" for both durations.
# precip = data['/Grid/precipitationCal'][:]

# # -- get rid of the dummy single-element first dimension,
# # transpose to get longitude on the x axis, and flip vertically
# # so that latitude is displayed south to north as it should be
# precip = np.flip( precip[0,:,:].transpose(), axis=0 )

# # -- display the precipitation data. Regions with missing data
# # values have negative values in the precip variable so allow
# # the color table to extend to negative values.
# import matplotlib.pyplot as plt
# plt.imshow( precip, vmin=-1, vmax=10, extent=[-180,180,-90,90] )

# # -- add a color bar
# cbar = plt.colorbar( )
# cbar.set_label('millimeters/hour')

# # -- display lat/lon grid lines
# for lon in np.arange(-90,90+1,90):
#   dummy = plt.plot( (lon,lon), (-90,+90), color="black", linewidth=1 )

# for lat in np.arange(-60,60+1,30):
#   dummy = plt.plot( (-180,+180), (lat,lat), color="black", linewidth=1 )

# # -- save the image to disk and display on the screen
# plt.savefig('imerg.png',dpi=200)
# plt.show()


    
fn = './plotting_data/3B-HHR-L.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V06B.HDF5' #filename (the ".h5" file)
with h5py.File(fn) as f:      
    # retrieve image data:
    precip = f['/Grid/precipitationCal'][:]
    print(f['/Grid/']['lat'])
    precip = np.flip( precip[0,:,:].transpose(), axis=0 )
    # get _FillValue for data masking
    #img_arr_fill = f[image].attrs['_FillValue'][0]   
    precip = np.ma.masked_less(precip, 0)
    precip = np.ma.masked_greater(precip, 200)
# retrieve extent of plot from file attributes:
    lower_lat = -90
    upper_lat = 90

# # retrieve attributes to calculate radiance from count:    
#     Sensor_Name = f.attrs['Sensor_Name'].decode('utf-8') 
#     img_fill = f[image].attrs['_FillValue'][0]   
#     img_inv  = f[image].attrs['invert'].decode('utf-8')
#     img_lrquad = f[image].attrs['lab_radiance_quad'][0]
#     img_lrscale = f[image].attrs['lab_radiance_scale_factor'][0]
#     img_lroff = f[image].attrs['lab_radiance_add_offset'][0]
# print('Done reading HDF5 file')  

## Use np.ma.masked_equal with integer values to  
## mask '_FillValue' data in corners:
#img_arr_m = np.ma.masked_equal(img_arr, img_arr_fill, copy=True)
img_arr_m = precip

map_proj = ccrs.Mercator()
# data_crs = ccrs.Geostationary(central_longitude=sat_long,
#                               satellite_height=sat_hght)

plt.figure(figsize=(10,10))
ax1 = plt.axes(projection=map_proj)
#ax1.coastlines()
#ax1.add_feature(cfeature.BORDERS, edgecolor='white', linewidth=0.5)
#ax1.gridlines(color='black', alpha=0.5, linestyle='--', linewidth=0.75, draw_labels=True)

map_proj_text = f'{str(type(map_proj)).split(".")[-1][:-2]}'
#data_crs_text = f'{str(type(data_crs)).split(".")[-1][:-2]}'
# plt.title(f'Plot1: Projection: {map_proj_text}\n' + \
#           f'Data Transform: {data_crs_text}\n' + \
#           f'\nRaster Data: {image} (masked)')
print('plotting data for Plot1 image')
im1 = ax1.imshow(img_arr_m, origin='upper', cmap='gray_r', vmin=0, vmax=10)
plt.colorbar(im1)
plt.show()