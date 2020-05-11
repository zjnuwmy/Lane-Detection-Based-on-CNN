import sys
import json
import os

from SuironVZ import visualize_data
#from file_finder import get_latest_filename

# Load image settings
with open('settings.json') as d:
    SETTINGS = json.load(d)

# Visualize latest filename
#filename = get_latest_filename() 

# If we specified which file
if len(sys.argv) > 1:
    filename = sys.argv[1]
#visualize_data("output_3.csv", width=SETTINGS['width'], height=SETTING
# S['height'], depth=SETTINGS['depth'], conf='./settings.json')

try:
    dir_len = len(os.listdir('D:/Temple University//RCDATA_CSV_7_15/'))
    for i in range(dir_len):
        visualize_data('D:/Temple University//RCDATA_CSV_7_15/output_%d.csv'%i, width=SETTINGS['width'], height=SETTINGS['height'], depth=SETTINGS['depth'], conf='./settings.json')

except (KeyboardInterrupt, SystemExit):
    raise 
except Exception as e:
   print(str(e))

print("complete")
