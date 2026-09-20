import numpy as np
import pandas as pd 
import csv
mortgage = pd.read_csv("File name")
#with open("data.csv", mode="r") as file:
#    reader = csv.DictReader(file)
#    coefficient = list(reader)
mortgage['Rent'] = 1
#'SITUS ZIP CODE' check this out
#Are we estimating average unit orbuilding
for each in mortgage['DEED SITUS ZIP CODE - STATIC'].unique():
    #tpool = mortgage['POOL INDICATOR']*coefficient[each][8]
    #tgarage = +int(mortgage['GARAGE CODE']!="000")*coefficient[each][7]
    #tbedrooms = +(mortgage['BEDROOMS - ALL BUILDINGS']/ mortgage['NUMBER OF UNITS'])*coefficient[each][0] #How would estimation work here, unit or building
    #tbathrooms = +mortgage['NUMBER OF BATHROOMS']*coefficient[each][1]
    #tsqft = +mortgage['UNIVERSAL BUILDING SQUARE FEET']*coefficient[each][2]
    #tapt =  +int(int(mortgage['PROPERTY INDICATOR CODE'])!=22)*coefficient[each][3]
    #tcomm = +int(int(mortgage['PROPERTY INDICATOR CODE'])!=20)*coefficient[each][4]
    #tcon =  +int(int(mortgage['PROPERTY INDICATOR CODE'])!=11)*coefficient[each][5]
    #tsfr =  +int(int(mortgage['PROPERTY INDICATOR CODE'])!=10)*coefficient[each][6]
    mortgage['Rent'] = np.where(mortgage['DEED SITUS ZIP CODE - STATIC'] == each,mortgage['POOL INDICATOR']*coefficient[each][8]+int(mortgage['GARAGE CODE']!="000")*coefficient[each][7]+(mortgage['BEDROOMS - ALL BUILDINGS']/ mortgage['NUMBER OF UNITS'])*coefficient[each][0]+mortgage['NUMBER OF BATHROOMS']*coefficient[each][1]+mortgage['UNIVERSAL BUILDING SQUARE FEET']*coefficient[each][2]+int(int(mortgage['PROPERTY INDICATOR CODE'])!=22)*coefficient[each][3]+int(int(mortgage['PROPERTY INDICATOR CODE'])!=20)*coefficient[each][4]+int(int(mortgage['PROPERTY INDICATOR CODE'])!=11)*coefficient[each][5] +int(int(mortgage['PROPERTY INDICATOR CODE'])!=10)*coefficient[each][6], mortgage['RENT'])

