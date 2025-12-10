import os
import pandas as pd 
import polars as pl
import numpy as np 
import sklearn as sc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import invwishart
from scipy.stats import invgamma
import math as math 
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from keras.utils import Sequence
from concurrent.futures import ThreadPoolExecutor
# import tensorflow as tf
# import keras
from sklearn.metrics import r2_score
import json
import gc
import matplotlib.pyplot as plt


cleaned = pd.read_csv("acleaned.csv")
def bayesianloop(sigma, sigmagroup, xtx, xty, beta, invlamb, betaindividual, bigsigma, xlist, ylist, xi,sigs,m,mu, p, size):
    siginv = np.linalg.inv(sigma)
    for j in range(m):
        cov = np.linalg.inv(siginv+(1/sigmagroup[j])*xtx[j].values)
        mean = cov@(siginv@beta+(1/sigmagroup[j])*xty[j].values)
        betaindividual[j] = np.random.multivariate_normal(mean, cov)
    covbeta = np.linalg.inv(invlamb+m*siginv)
    sum_betas = np.sum(betaindividual, axis=0)
    mubeta = covbeta@(invlamb@mu+siginv@sum_betas)
    beta = np.random.multivariate_normal(mubeta, covbeta)
    summy = bigsigma.copy()
    for j in range(m):
        a = betaindividual[j]-beta
        summy+=np.outer(a, a)
    sigma = invwishart.rvs(df=p+2+m, scale=summy)
    for j in range(m):
        second = ylist[j]-xlist[j].values@betaindividual[j]
        t = [x ** 2 for x in second]
        secondary = np.sum(np.array(t))
        sigmagroup[j] = invgamma.rvs(a =.5*(1+size[j]), scale = .5*(xi+secondary) , size =1)[0]
    inversesigma =  (1/sigs)+.5*np.sum(1/np.array(sigmagroup))
    xi = np.random.gamma(1+.5*m,1/inversesigma)
    return sigma, beta, betaindividual, xi, sigmagroup
def msecalc(betaindividual1,xlisttest,ylisttest):
    ypred = []
    for j in range(m):
        ypred.append(xlisttest[j]@betaindividual1[j])
        ypred[j] = np.array(ypred[j])
        ylisttest[j] = np.array(ylisttest[j])
    count = 0
    msetemp = 0 
    for j in range(m):
        for i in range(len(ypred[j])):
            msetemp+= (ypred[j][i]-ylisttest[j][i])**2
            count+=1
    mse = (1/count)*msetemp
    return mse
def rhat(lister, num):
    n = len(lister[0])
    mean = []
    sdchain = []
    for i in range(num):
        mean.append(np.mean(np.array(lister[i])))
        sdchain.append(np.var(np.array(lister[i]), ddof=1))
    B = n*np.var(np.array(mean),ddof=1)
    W = np.mean(sdchain)
    varpsiy = ((n-1)/n)*W+(1/n)*B
    rhat = np.sqrt(varpsiy/W)
    return rhat
def maxrhat(lister,num,param):
    maxval = 0
    for i in range(len(lister)):
            a = rhat(lister[i],num)
            if a>maxval:
                maxval = a
    return maxval
def listinitializer(betaindividual1, betaindividual2, betaindividual3,betaindividual4):
    lister = []
    for i in range(len(betaindividual1)):
        for j in range(len(betaindividual1[0])):
            lister.append([[betaindividual1[i][j]],[betaindividual2[i][j]], [betaindividual3[i][j]], [betaindividual4[i][j]] ])
    return lister
def listappender(lister,param,betaindividual1, betaindividual2, betaindividual3,betaindividual4,m):
    for i in range(m):
        for j in range(param):
            z = i*param+j
            lister[z][0].append(betaindividual1[i][j])
            lister[z][1].append(betaindividual2[i][j])
            lister[z][2].append(betaindividual3[i][j])
            lister[z][3].append(betaindividual4[i][j])
    return lister

cleaned['INTERCEPT'] = 1
t = ['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'ZIP', 'INTERCEPT']

mu = np.array([
    -1.00774303e-01,  1.67737087e-01,  4.58936502e-04,  9.81264571e-03, 
    -2.93593061e-02,  1.29088669e-01, -8.38429325e-02,  1.52842902e-01, 
    -8.40472908e-02,  6.51888483e+00
])

sigs = 0.1833664 

lamb = np.array([
    [ 6.20035246e+00, -3.58903197e+00, -4.51962845e-03,  5.72635667e-01,  2.69108986e+00,  5.88809435e-01, -2.42599313e+00,  2.47122921e-01,  1.15872015e-01, -1.38180939e+00],
    [-3.58903197e+00,  1.25802700e+01, -7.32749460e-03,  1.15797905e+00,  2.73554620e+00, -1.50989972e-01,  3.78100893e+00, -1.11963954e-01, -3.80338954e-01, -6.25751273e+00],
    [-4.51962845e-03, -7.32749460e-03,  2.25171969e-05,  1.28166905e-03, -6.73213789e-03,  9.82066580e-04, -4.41129577e-03, -4.79984530e-04,  7.31604797e-05, -4.90178389e-03],
    [ 5.72635667e-01,  1.15797905e+00,  1.28166905e-03,  4.16213837e+01,  3.84308200e+01,  3.84443713e+01,  3.65729396e+01, -6.83973916e-01, -8.72583272e-01, -4.27492601e+01],
    [ 2.69108986e+00,  2.73554620e+00, -6.73213789e-03,  3.84308200e+01,  1.74212185e+03,  3.80719223e+01,  3.81990500e+01, -7.12978347e-02,  7.00553569e-01, -4.06496000e+01],
    [ 5.88809435e-01, -1.50989972e-01,  9.82066580e-04,  3.84443713e+01,  3.80719223e+01,  1.00679003e+02,  3.70421028e+01,  1.29626583e+00, -2.42306366e-01, -4.01466318e+01],
    [-2.42599313e+00,  3.78100893e+00, -4.41129577e-03,  3.65729396e+01,  3.81990500e+01,  3.70421028e+01,  5.25767088e+01,  6.30170310e-01,  1.46100043e+00, -3.43151619e+01],
    [ 2.47122921e-01, -1.11963954e-01, -4.79984530e-04, -6.83973916e-01, -7.12978347e-02,  1.29626583e+00,  6.30170310e-01,  4.83268903e+01, -4.85717321e+00,  1.19654995e-01],
    [ 1.15872015e-01, -3.80338954e-01,  7.31604797e-05, -8.72583272e-01,  7.00553569e-01, -2.42306366e-01,  1.46100043e+00, -4.85717321e+00,  9.36057673e+00, -2.05063007e+00],
    [-1.38180939e+00, -6.25751273e+00, -4.90178389e-03, -4.27492601e+01, -4.06496000e+01, -4.01466318e+01, -3.43151619e+01,  1.19654995e-01, -2.05063007e+00,  5.97457038e+01]
])
invlamb = np.linalg.inv(lamb)
sigma = lamb
bigsigma = lamb
beta = mu
xi = math.sqrt(sigs)
t = ['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'INTERCEPT']
print("Linear done")
listofzips = cleaned['ZIP'].unique()
m = len(listofzips)
p = mu.shape[0]+1
print(p)
betaindividual = []
ylist= []
xlist = []
xlisttest = []
ylisttest = [] 
xtx = []
xty = []
size =[]
for each in listofzips:
    xyz = cleaned[cleaned['ZIP'] == each]
    X = xyz[t]
    if X.shape[0]<5:
        continue
    y = np.array(xyz['logrent'])
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = 24)
    n = X.shape[0]
    size.append(n)
    xlist.append(Xtrain)
    ylist.append(ytrain)
    xlisttest.append(Xtest)
    ylisttest.append(ytest)
    xtx.append(Xtrain.T@Xtrain)
    xty.append(Xtrain.T@ytrain)

print("Bayesian time!")


gc.collect()
xi1 = np.random.gamma(1, (1/sigs))
xi2 = np.random.gamma(1, (1/sigs))
xi3 = np.random.gamma(1, (1/sigs))
xi4 = np.random.gamma(1, (1/sigs))
sigmagroup1 = invgamma.rvs(a =.5, scale = .5*xi1, size =m)
sigmagroup2 = invgamma.rvs(a =.5, scale = .5*xi2, size =m)
sigmagroup3 = invgamma.rvs(a =.5, scale = .5*xi3, size =m)
sigmagroup4 = invgamma.rvs(a =.5, scale = .5*xi4, size =m)

sigma1 = invwishart.rvs(df=10, scale=bigsigma)
sigma2 = invwishart.rvs(df=10, scale=bigsigma)
sigma3 = invwishart.rvs(df=10, scale=bigsigma)
sigma4 = invwishart.rvs(df=10, scale=bigsigma)
beta1 = np.random.multivariate_normal(mean = mu, cov = lamb)
beta2 = np.random.multivariate_normal(mean = mu, cov = lamb)
beta3 = np.random.multivariate_normal(mean = mu, cov = lamb)
beta4 = np.random.multivariate_normal(mean = mu, cov = lamb)
betaindividual1 = []
betaindividual2 = []
betaindividual3 = []
betaindividual4 = []
for j in range(m):
    betaindividual1.append(np.random.multivariate_normal(mean = beta1, cov = sigma1))
    betaindividual2.append(np.random.multivariate_normal(mean = beta2, cov = sigma2))
    betaindividual3.append(np.random.multivariate_normal(mean = beta3, cov = sigma3))
    betaindividual4.append(np.random.multivariate_normal(mean = beta4, cov = sigma4))
print("Initialize done")

for i in range(100):
    print(i)
    sigma1, beta1, betaindividual1, xi1, sigmagroup1 = bayesianloop(sigma1, sigmagroup1, xtx, xty, beta1, invlamb, betaindividual1, bigsigma, xlist, ylist, xi1,sigs, m,mu,p,size)
    sigma2, beta2, betaindividual2, xi2, sigmagroup2 = bayesianloop(sigma2, sigmagroup2, xtx, xty, beta2, invlamb, betaindividual2, bigsigma, xlist, ylist, xi2,sigs,m,mu,p,size)
    sigma3, beta3, betaindividual3, xi3, sigmagroup3 = bayesianloop(sigma3, sigmagroup3, xtx, xty, beta3, invlamb, betaindividual3, bigsigma, xlist, ylist, xi3,sigs,m,mu,p,size)
    sigma4, beta4, betaindividual4, xi4, sigmagroup4 = bayesianloop(sigma4, sigmagroup4, xtx, xty, beta4, invlamb, betaindividual4, bigsigma, xlist, ylist, xi4,sigs,m,mu,p,size)
print("Sampling done")
maxrhats = []

totallist = listinitializer(betaindividual1,betaindividual2, betaindividual3, betaindividual4)
r = 100000
i=1
while r>1.3:
    sigma1, beta1, betaindividual1, xi1, sigmagroup1 = bayesianloop(sigma1, sigmagroup1, xtx, xty, beta1, invlamb, betaindividual1, bigsigma, xlist, ylist, xi1,sigs,m,mu,p,size)
    sigma2, beta2, betaindividual2, xi2, sigmagroup2 = bayesianloop(sigma2, sigmagroup2, xtx, xty, beta2, invlamb, betaindividual2, bigsigma, xlist, ylist, xi2,sigs,m,mu,p,size)
    sigma3, beta3, betaindividual3, xi3, sigmagroup3 = bayesianloop(sigma3, sigmagroup3, xtx, xty, beta3, invlamb, betaindividual3, bigsigma, xlist, ylist, xi3,sigs,m,mu,p,size)
    sigma4, beta4, betaindividual4, xi4, sigmagroup4 = bayesianloop(sigma4, sigmagroup4, xtx, xty, beta4, invlamb, betaindividual4, bigsigma, xlist, ylist, xi4,sigs,m,mu,p,size)
    totallist = listappender(totallist,10,betaindividual1, betaindividual2, betaindividual3,betaindividual4,m)
    if i%10 == 0:
        r = maxrhat(totallist,num = 4, param = 10)
        maxrhats.append(r)
        print(r)
    i+=1
print("convergence done")
print("MSE:"+str(msecalc(betaindividual4,xlisttest, ylisttest)))
print("MSE:"+str(msecalc(betaindividual3,xlisttest, ylisttest)))
print("MSE:"+str(msecalc(betaindividual2,xlisttest, ylisttest)))
print("MSE:"+str(msecalc(betaindividual1,xlisttest, ylisttest)))
finallist = []
for i in range(30):
    sigma1, beta1, betaindividual1, xi1, sigmagroup1 = bayesianloop(sigma1, sigmagroup1, xtx, xty, beta1, invlamb, betaindividual1, bigsigma, xlist, ylist, xi1,sigs,m,mu,p,size)
    sigma2, beta2, betaindividual2, xi2, sigmagroup2 = bayesianloop(sigma2, sigmagroup2, xtx, xty, beta2, invlamb, betaindividual2, bigsigma, xlist, ylist, xi2,sigs,m,mu,p,size)
    sigma3, beta3, betaindividual3, xi3, sigmagroup3 = bayesianloop(sigma3, sigmagroup3, xtx, xty, beta3, invlamb, betaindividual3, bigsigma, xlist, ylist, xi3,sigs,m,mu,p,size)
    sigma4, beta4, betaindividual4, xi4, sigmagroup4 = bayesianloop(sigma4, sigmagroup4, xtx, xty, beta4, invlamb, betaindividual4, bigsigma, xlist, ylist, xi4,sigs,m,mu,p,size)
    finallist.append(betaindividual1)
    finallist.append(betaindividual2)
    finallist.append(betaindividual3)
    finallist.append(betaindividual4)
print("final mean")
finalbetalist = np.mean(finallist, axis=0)
print("Final MSE:"+str(msecalc(finalbetalist,xlisttest, ylisttest)))