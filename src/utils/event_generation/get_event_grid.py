import csv
import numpy as np

def unique(lakes):
    lakes = np.asarray(lakes)[:,0:2]
    return np.unique(lakes,axis=0)

event_path = './events/floods/'
event_filenames = ['flow_events_50.csv','one_year_floods.csv']

locations = []
for filename in event_filenames:
    with open(event_path+filename,newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 5:
                i=i+1
                continue
            row = [float(i) for i in row]
            row = [row[0],row[1]]
            locations.append(row)

locations = unique(locations)
with open(event_path+'flood_event_points.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]'])
    for loc in locations:
        csvwriter.writerow(loc)