import csv

def sub_sort(sub_li):
 
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key = lambda x: x[3],reverse=True)
    return sub_li

filename = './events/lakes/HydroLAKES_points_v10.csv'

lake_locations = []
with open(filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 5:
            i=i+1
            continue
        row = [float(i) for i in row]
        lake_locations.append(row)


lake_locations_sorted = sub_sort(lake_locations)
with open('./events/lakes/lake_event_points_reduced.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]'])
    i = 0
    for loc in lake_locations_sorted:
        location = [loc[1],loc[2]]
        csvwriter.writerow(location)
        i = i+1
        if i > 100:
            break