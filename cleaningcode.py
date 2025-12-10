import pandas as pd 
import numpy as np 
import sklearn as sc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import invwishart
from scipy.stats import invgamma
import math as math 
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score
import gc
import matplotlib.pyplot as plt



#CLEANING
sample = pd.read_csv("a.csv")
samps = sample.drop(['GRANITE', 'STAINLESS','GYM','DOORMAN','FURNISHED','LAUNDRY', 'CLUBHOUSE','LATITUDE','LONGITUDE','DESCRIPTION', 'GARAGE_COUNT','ADDRESS', 'COMPANY','ID','NEIGHBORHOOD','SCRAPED_TIMESTAMP','YEAR_BUILT','AVAILABLE_AT','AVAILABILITY_STATUS','ID'], axis=1)
samps['YEAR'] =pd.to_datetime(samps['DATE_POSTED'])
samps['MONTH'] = pd.to_numeric(samps['YEAR'].dt.month)
samps['YEAR'] = pd.to_numeric(samps['YEAR'].dt.year)-2014

#samps['MONTH'] = pd.to_numeric(pd.to_datetime(samps['DATE_POSTED'],dayfirst=True, format = "%m"))
samps['TIME'] = 12*samps['YEAR']-1+samps['MONTH']
one = pd.get_dummies(samps,columns = ['BUILDING_TYPE'], drop_first=False)
new_samps = pd.get_dummies(one,columns = ['GARAGE','POOL'], drop_first=True)
zen = len(samps['ZIP'].unique())
new_samps.drop(columns = [ 'BUILDING_TYPE_TIME', 'BUILDING_TYPE_MH', 'BUILDING_TYPE_TH','DATE_POSTED','YEAR','MONTH'],axis =1,inplace=True)
new_samps["logrent"] = np.log(new_samps["RENT_PRICE"])
new_samps = new_samps.dropna()
zip_counts = new_samps.groupby("ZIP")["ZIP"].transform("count")
# cleaned = samps[zip_counts > 30]
cleaned = new_samps[zip_counts > 50]
del samps
del new_samps
gc.collect()
cols = [
    "BUILDING_TYPE_APT", "BUILDING_TYPE_COMM", "BUILDING_TYPE_CON",
    "BUILDING_TYPE_SFR", "GARAGE_Y", "POOL_Y"
]
cleaned[cols] = cleaned[cols].replace(
    {"True": 1, "False": 0, "true": 1, "false": 0}
).astype(int)

cleaned.to_csv("acleaned.csv")
print("Data cleaning is done")