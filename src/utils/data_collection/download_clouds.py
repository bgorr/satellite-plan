from goespy.Downloader import ABI_Downloader

destination_path = './clouds/'
bucket = "noaa-goes16"
year="2020"
month="01"
day='01'
hour='11'
product='ABI-L2-ACMF'
channel='C13'

ABI_Downloader(destination_path,bucket,year,month,day,hour,product,channel)