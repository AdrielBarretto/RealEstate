import pandas as pd 
import numpy as np 
import sklearn as sc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack
import statsmodels.api as sm
import matplotlib.pyplot as plt
def apca(numb,a):
    ab = a.sparse.to_coo().tocsr()
    pca = sc.decomposition.TruncatedSVD(200)
    z = pca.fit_transform(ab)
    explained_variance_ratios = pca.explained_variance_ratio_
    cumulative_variance_explained = explained_variance_ratios.cumsum()
    d = pd.DataFrame(z, index = cleaned.index)
    pca_set = cleaned.join(d)
    colpca = ['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y','TIME']+d.columns.to_list()
    Xpca = pca_set[colpca]
    Xpca.columns = Xpca.columns.astype(str)
    X_trainpca, X_testpca, y_trainpca, y_testpca = train_test_split(Xpca, pca_set['logrent'], test_size=0.2, random_state =24)
    y_testpca = np.array(y_testpca)
    model = LinearRegression()
    model.fit(X_trainpca, y_trainpca)
    y_predpca = model.predict(X_testpca)
    msepca = 0
    msepca = np.mean((y_testpca-y_predpca)**2)
    return msepca, cumulative_variance_explained
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
cleaned = new_samps.groupby('ZIP').filter(lambda x: len(x) > 30)
cleaned.dropna(inplace = True)
listofzips = cleaned['ZIP'].unique()
print(len(listofzips))
a = pd.get_dummies(cleaned['ZIP'],drop_first = False, sparse=True)
mselist = [0]
cumvariance = [0]
for i in range(1,200):
    print(i)
    mse, cvar = apca(i, a)
    mselist.append(mse)
    cumvariance.append(cvar)
plt.scatter(cumvariance, mselist)
plt.xlabel("Cumulative Variance")
plt.xlabel("MSE")
plt.savefig("MSEcumsum.png")
model = LinearRegression()
model.fit(cumvariance,mselist)
y = np.array(mselist)
X = np.array(cumvariance)
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const)
results = model.fit()
print(results.summary())
