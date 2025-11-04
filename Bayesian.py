import pandas as pd 
import numpy as np 
import sklearn as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import keras
import gc
import matplotlib.pyplot as plt
from scipy.stats import invwishart
from scipy.stats import invgamma
import math as math 
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
cleaned = new_samps.groupby('ZIP').filter(lambda x: len(x) > 25)
a = cleaned[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'ZIP']].fillna(0)
b = np.array(cleaned['logrent'])
#split data
Xtrain, X_test, y, y_test = train_test_split(a, b, test_size=0.2, random_state = 24)
train = pd.DataFrame(y, columns=['logrent'])
traindf = pd.concat([Xtrain, train], axis=1)
test = pd.DataFrame(y_test, columns=['logrent'])
testdf = pd.concat([X_test, test], axis=1)
#run big linear
X = Xtrain[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y']]
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mu = model.intercept_+model.coef_
na = X.shape[0]
p = X.shape[1]
f = np.sum((y-y_pred)**2)
sig = f/(na-p-1)
lamb = na*p*sig*np.linalg.inv(X.T@X)
invlamb = np.linalg.inv(lamb)
sigma = lamb
bigsigma = lamb
beta = mu
xi = math.sqrt(sig)


listofzips = a['ZIP'].unique()
m = len(listofzips)
betaindividual = []
ylist= []
xlist = []
xlisttest = []
ylisttest = [] 
xtx = []
xty = []
sigmagroup = []
size =[]
for each in listofzips:
    betaa = []
    xyz = traindf[traindf['ZIP'] == each]
    X = xyz[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y']]
    xlist.append(X)
    y = np.array(xyz['logrent'])
    ylist.append(y)
    abc = testdf[testdf['ZIP'] == each]
    a = abc[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y']]
    b = np.array(abc['logrent'])
    xlisttest.append(a)
    ylisttest.append(b)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    n = X.shape[0]
    size.append(n)
    p = X.shape[1]
    f = np.sum((y-y_pred)**2)
    sig = f/(n-p-1)
    sigmagroup.append(sig)
    betaa = model.intercept_+model.coef_
    betaindividual.append(betaa)
    xtx.append(X.T@X)
    xty.append(X.T@y)
#intercept?
convergence_rate = 40000000000
for i in range(convergence_rate):
    siginv = np.linalg.inv(sigma)
    for j in range(m):
        cov = np.linalg.inv(siginv+(1/sigmagroup[j])*xtx[j])
        mean = cov@(siginv@beta+(1/sigmagroup[j])*xty[j])
        betaindividual[j] = np.random.multivariate_normal(mean, cov, size =1)
    covbeta = np.linalg.inv(invlamb+m*siginv)@(invlamb@mu+siginv)
    mubeta = covbeta@(invlamb@mu+siginv@np.sum(betaindividual))
    beta = np.random.multivariate_normal(mubeta, covbeta, size =1)
    summy = bigsigma
    for j in range(m):
        a = betaindividual[j]-beta
        summy+= a@a.T
    scale = np.linalg.inv(summy)
    sigma = invwishart.rvs(df=p+2+m, scale=scale, size=1)
    for j in range(m):
        second = ylist[j]-xlist[j]@betaindividual[j]
        t = [x ** 2 for x in second]
        secondary = np.sum(np.array(t))
        sigmagroup[j] = invgamma(alpha =.5*(1+size[j]), beta = .5*(xi+secondary) , size =1)
    xi = np.random.gamma(1+.5*m, (1/sig)+.5*np.sum(np.array(sigmagroup)))
ypred = []
for j in range(m):
    ypred.append(xlisttest[j]@betaindividual[j])
t = np.array(ylisttest-ypred)
final = [x ** 2 for x in t]
mse = (1/na)*np.sum(np.sum(t))
print(mse)