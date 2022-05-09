import os
import re
import csv

with open("C:/Users/Marc/Desktop/tuh_eeg_events/v1.0.1/edf_list.csv",
            "a", newline='') as f:
    write = csv.writer(f)
    for root, dirs, files in os.walk("C:/Users/Marc/Desktop/tuh_eeg_events/v1.0.1/edf"):
        for file in files:
            if file.endswith(".edf"):
                root = re.sub(r"\\", "/", root)
                m = re.search('edf/(.+)', root)
                path = '../edf/' + m.group(1) + "/" + file
                write.writerow([path])

with open("C:/Users/Marc/Desktop/tuh_eeg_events/v1.0.1/rec_list.csv",
            "a", newline='') as f:
    write = csv.writer(f)
    for root, dirs, files in os.walk("C:/Users/Marc/Desktop/tuh_eeg_events/v1.0.1/edf"):
        for file in files:
            if file.endswith(".rec"):
                root = re.sub(r"\\", "/", root)
                m = re.search('edf/(.+)', root)
                path = '../edf/' + m.group(1) + "/" + file
                write.writerow([path])
