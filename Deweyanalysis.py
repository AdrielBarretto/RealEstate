import pandas as pd 
import numpy as np 

sample = pd.read_csv("a.csv")
samps = sample.drop(['GRANITE', 'STAINLESS','GYM','DOORMAN','FURNISHED','LAUNDRY', 'CLUBHOUSE','LATITUDE','LONGITUDE','DESCRIPTION', 'GARAGE_COUNT','ADDRESS', 'COMPANY','ID','NEIGHBORHOOD','SCRAPED_TIMESTAMP','YEAR_BUILT','AVAILABLE_AT','AVAILABILITY_STATUS','ID'], axis=1)
samps['YEAR'] =pd.to_datetime(samps['DATE_POSTED'])
samps['MONTH'] = pd.to_numeric(samps['YEAR'].dt.month)
samps['YEAR'] = pd.to_numeric(samps['YEAR'].dt.year)-2014

#samps['MONTH'] = pd.to_numeric(pd.to_datetime(samps['DATE_POSTED'],dayfirst=True, format = "%m"))
samps['TIME'] = 12*samps['YEAR']-1-samps['MONTH']
one = pd.get_dummies(samps,columns = ['BUILDING_TYPE'], drop_first=False)
two = pd.get_dummies(samps,columns = ['GARAGE','POOL'], drop_first=True)
zen = len(samps['ZIP'].unique())
new_samps = pd.concat([samps,one,two],axis=1)
new_samps.drop(columns = [ 'BUILDING_TYPE_TIME', 'BUILDING_TYPE_MH', 'BUILDING_TYPE_TH','BUILDING_TYPE','GARAGE','POOL','DATE_POSTED','YEAR','MONTH'],axis =1,inplace=True)
q = np.sum(np.array(samps['ZIP'].value_counts().to_list())>25)/zen
r = np.sum(np.array(samps['ZIP'].value_counts().to_list())>70)/zen
s = np.sum(np.array(samps['ZIP'].value_counts().to_list())>40)/zen
t = np.sum(np.array(samps['ZIP'].value_counts().to_list())>50)/zen
print(q)
print(r)
print(s)
print(t)


#filter by number of entries
#0-1 columns for neural network 
#MCA 
#Straight eregression 
#VAR 
#Bayesian Hierarchical