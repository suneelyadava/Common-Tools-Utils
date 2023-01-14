import csv
import json

csv_file = open('Top 5000 Incidents-dex.csv', 'r')
json_file = open('Top 5000 Incidents-dex.json', 'w')

fieldnames = ("ID", "Title")
reader = csv.DictReader( csv_file, fieldnames)
out = json.dumps([row for row in reader])
json_file.write(out)
