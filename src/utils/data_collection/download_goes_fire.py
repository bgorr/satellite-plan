from goes2go import GOES
import netCDF4

G = GOES(satellite=17, product="ABI-L2-FDCF", domain='F')

G.timerange(start='2022-02-02 00:00', end='2022-02-03 00:00')