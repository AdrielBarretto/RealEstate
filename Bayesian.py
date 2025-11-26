import pandas as pd 
import numpy as np 
import sklearn as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import invwishart
from scipy.stats import invgamma
import math as math 
#Bayesian loop samples
#MSE Calc does mse calcs
#Rhat calculates rubin gelman statistic
#Maxrhat gets the highest rhat
#Listinitializer sets up the list of rhats 
#Listappender adds it 
def bayesianloop(sigma, sigmagroup, xtx, xty, beta, invlamb, betaindividual, bigsigma, xlist, ylist, xi,sigs,m):
    siginv = np.linalg.inv(sigma)
    for j in range(m):
        cov = np.linalg.inv(siginv+(1/sigmagroup[j])*xtx[j].values)
        mean = cov@(siginv@beta+(1/sigmagroup[j])*xty[j].values)
        betaindividual[j] = np.random.multivariate_normal(mean, cov)
    covbeta = np.linalg.inv(invlamb+m*siginv)
    sum_betas = np.sum(betaindividual, axis=0)
    mubeta = covbeta@(invlamb@mu+siginv@sum_betas)
    beta = np.random.multivariate_normal(mubeta, covbeta)
    summy = bigsigma
    for j in range(m):
        a = betaindividual[j]-beta
        summy+=np.outer(a, a)
    scale = np.linalg.inv(summy)
    sigma = invwishart.rvs(df=p+2+m, scale=scale)
    for j in range(m):
        second = ylist[j]-xlist[j].values@betaindividual[j]
        t = [x ** 2 for x in second]
        secondary = np.sum(np.array(t))
        sigmagroup[j] = invgamma.rvs(a =.5*(1+size[j]), scale = .5*(xi+secondary) , size =1)[0]
    xi = np.random.gamma(1+.5*m, (1/sigs)+.5*np.sum(np.array(sigmagroup)))
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
def rhat(list, num):
    n = len(list[0])
    mean = []
    sdchain = []
    for i in range(num):
        mean.append(np.mean(np.array(list[i])))
        sdchain.append(np.var(np.array(list[i]), ddof=1))
    fullmean = np.mean(np.array(mean))
    B = n*np.var(np.array(sdchain),ddof=1)
    W = np.mean(sdchain)
    varpsiy = ((n-1)/n)*W+(1/n)*B
    rhat = np.sqrt(varpsiy/W)
    return rhat
def maxrhat(lister,num,param):
    maxval = 0
    for i in range(len(lister)):
            a = rhat(lister[i],4)
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
'''
Testing
a = [[1,2],[1,2], [1,2], [1,2]]
b = [[3,4],[3,4], [3,4], [3,4]]
c = [[5,6],[5,6], [5,6], [5,6]]
d = [[7,8], [7,8], [7,8], [7,8]]
q = listinitializer(a,b,c,d)
print(q)
q = listappender(q,2,a,b,c,d,4)
print(q)
print(len(q))
'''
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
cleaned = cleaned.groupby('ZIP').filter(lambda x: len(x) > 30)
#Same cleaning 
t = ['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'ZIP','logrent']
cleaned = cleaned[t].replace({True:1,False:0})
cleaned['INTERCEPT'] = 1
t = ['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'ZIP', 'INTERCEPT']
#b = np.array(cleaned['logrent'])
print("Cleaning done")
#split data
na = cleaned[t].shape[0]
p =  cleaned[t].shape[1]
'''
Xtrain, X_test, y, ytest= train_test_split(cleaned[t], b, test_size=0.2, random_state = 24)
X = Xtrain[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'INTERCEPT']]
model = LinearRegression(fit_intercept =False)
model.fit(X, y)
y_pred = model.predict(X)
mu = model.coef_.flatten()
print(mu)
na = X.shape[0]
p = X.shape[1]
f = np.sum((y-y_pred)**2)
sigs = f/(na-p-1)
print(sigs)
X1=X.values
lamb = (na*p*sigs)*np.linalg.inv(X1.T@X1)
'''
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
#Initial sampling/Burn in 
for i in range(500):
    sigma1, beta1, betaindividual1, xi1, sigmagroup1 = bayesianloop(sigma1, sigmagroup1, xtx, xty, beta1, invlamb, betaindividual1, bigsigma, xlist, ylist, xi1,sigs, m)
    sigma2, beta2, betaindividual2, xi2, sigmagroup2 = bayesianloop(sigma2, sigmagroup2, xtx, xty, beta2, invlamb, betaindividual2, bigsigma, xlist, ylist, xi2,sigs,m)
    sigma3, beta3, betaindividual3, xi3, sigmagroup3 = bayesianloop(sigma3, sigmagroup3, xtx, xty, beta3, invlamb, betaindividual3, bigsigma, xlist, ylist, xi3,sigs,m)
    sigma4, beta4, betaindividual4, xi4, sigmagroup4 = bayesianloop(sigma4, sigmagroup4, xtx, xty, beta4, invlamb, betaindividual4, bigsigma, xlist, ylist, xi4,sigs,m)
print("Sampling done")
maxrhats = []

totallist = listinitializer(betaindividual1,betaindividual2, betaindividual3, betaindividual4)
r = 100000
while r>1.05:
    sigma1, beta1, betaindividual1, xi1, sigmagroup1 = bayesianloop(sigma1, sigmagroup1, xtx, xty, beta1, invlamb, betaindividual1, bigsigma, xlist, ylist, xi1,sigs,m)
    sigma2, beta2, betaindividual2, xi2, sigmagroup2 = bayesianloop(sigma2, sigmagroup2, xtx, xty, beta2, invlamb, betaindividual2, bigsigma, xlist, ylist, xi2,sigs,m)
    sigma3, beta3, betaindividual3, xi3, sigmagroup3 = bayesianloop(sigma3, sigmagroup3, xtx, xty, beta3, invlamb, betaindividual3, bigsigma, xlist, ylist, xi3,sigs,m)
    sigma4, beta4, betaindividual4, xi4, sigmagroup4 = bayesianloop(sigma4, sigmagroup4, xtx, xty, beta4, invlamb, betaindividual4, bigsigma, xlist, ylist, xi4,sigs,m)
    if i%10 == 0:
        totallist = listappender(totallist,10,betaindividual1, betaindividual2, betaindividual3,betaindividual4,m)
        r = maxrhat(totallist,num = 4, param = 10)
        maxrhats.append(r)
        print(r)
    i+=1

print("MSE:"+str(msecalc(betaindividual4,xlisttest, ylisttest)))