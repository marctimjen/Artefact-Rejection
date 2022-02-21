
import csv

#with open("C:/Users/Bruger/Documents/Uni/6. Semester/BP/data/v2.1.0/lists/edf_01_tcp_ar.list", "r") as file: # Læser csv-fil
#    list = csv.reader(file)
#    for rows in list:
#        print(rows[0])

with open("C:/Users/Bruger/Documents/GitHub/edf/01_tcp_ar/002/00000254/s005_2010_11_15/00000254_s005_t000.rec", "r") as file: # Læser csv-fil
    list = csv.reader(file)
    for rows in list:
        print(rows)
