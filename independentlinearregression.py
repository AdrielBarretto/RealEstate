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


cleaned = pd.read_csv("acleaned.csv")
listofzips = cleaned["ZIP"].unique()
print("Number of zip codes"+str(len(listofzips)))
mse_list  = []
ct = 0
for each in listofzips:
    ct += 1
    if ct%100 == 0:
        print("Zip code "+str(each))
    gc.collect()
    cleaned1 = cleaned[cleaned['ZIP'] == each]
    X = cleaned1[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y','TIME']].fillna(0)
    #X.dropna(inplace = True)
    if X.shape[0] < 5:
        continue
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, cleaned1['logrent'], test_size=0.2)
        y_test = np.array(y_test)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = 0
        for i in range(len(y_test)):
            mse+= (y_test[i]-y_pred[i])**2
        mse = mse/len(y_test)
        mse_list.append(mse)
average = np.mean(mse_list)
print("MSE for Independent Linear Regressions: "+str(average))