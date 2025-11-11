import pandas as pd 
import numpy as np 
import sklearn as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import invwishart
from scipy.stats import invgamma
import math as math 

def bayesianloop(sigma, sigmagroup, xtx, xty, beta, invlamb, betaindividual, bigsigma, xlist, ylist, xi,sigs):
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
a = cleaned[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'ZIP']]
a=a.replace({True:1,False:0})
a['INTERCEPT'] = 1
b = np.array(cleaned['logrent'])
print("Cleaning done")
#split data
c = pd.DataFrame(b, columns=['logrent'])
XYZ = pd.concat([a.reset_index(drop=True),c], axis=1)
Xtrain, X_test, y, y_test = train_test_split(a, b, test_size=0.2, random_state = 24)
#run big linear
X = Xtrain[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'INTERCEPT']]
model = LinearRegression(fit_intercept =False)
model.fit(X, y)
y_pred = model.predict(X)
mu = model.coef_.flatten()
na = X.shape[0]
p = X.shape[1]
f = np.sum((y-y_pred)**2)
sigs = f/(na-p-1)
X1=X.values
lamb = (na*p*sigs)*np.linalg.inv(X1.T@X1)
invlamb = np.linalg.inv(lamb)
sigma = lamb
bigsigma = lamb
beta = mu
xi = math.sqrt(sigs)

print("Linear done")
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
    xyz = XYZ[XYZ['ZIP'] == each]
    X = xyz[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y', 'INTERCEPT']]
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


convergence_rate = 5
count = 0 
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
 

r = 1000
listbeta11 = [[betaindividual1[1][1]],[betaindividual2[1][1]], [betaindividual3[1][1]], [betaindividual1[1][1]]]
listbeta105 = [[betaindividual1[10][5]], [betaindividual2[10][5]], [betaindividual2[10][5]], [betaindividual2[10][5]]]
i=1
while r>1.1:
   print(i)
   sigma1, beta1, betaindividual1, xi1, sigmagroup1 = bayesianloop(sigma1, sigmagroup1, xtx, xty, beta1, invlamb, betaindividual1, bigsigma, xlist, ylist, xi1,sigs)
   listbeta11[0].append(betaindividual1[1][1])
   listbeta105[0].append(betaindividual1[10][5])
   print(msecalc(betaindividual1,xlisttest, ylisttest))
   sigma2, beta2, betaindividual2, xi2, sigmagroup2 = bayesianloop(sigma2, sigmagroup2, xtx, xty, beta2, invlamb, betaindividual2, bigsigma, xlist, ylist, xi2,sigs)
   listbeta11[1].append(betaindividual2[1][1])
   listbeta105[1].append(betaindividual2[10][5])
   print(msecalc(betaindividual2,xlisttest, ylisttest))
   sigma3, beta3, betaindividual3, xi3, sigmagroup3 = bayesianloop(sigma3, sigmagroup3, xtx, xty, beta3, invlamb, betaindividual3, bigsigma, xlist, ylist, xi3,sigs)
   listbeta11[2].append(betaindividual3[1][1])
   listbeta105[2].append(betaindividual3[10][5])
   print(msecalc(betaindividual3,xlisttest, ylisttest))
   sigma4, beta4, betaindividual4, xi4, sigmagroup4 = bayesianloop(sigma4, sigmagroup4, xtx, xty, beta4, invlamb, betaindividual4, bigsigma, xlist, ylist, xi4,sigs)
   listbeta11[3].append(betaindividual4[1][1])
   listbeta105[3].append(betaindividual4[10][5])
   print(msecalc(betaindividual4,xlisttest, ylisttest))
   r = rhat(listbeta11, 4)
   print(r)
   i+=1
   #print(str(r)+" Individual: "+str(rhat(listbeta11, 4))+ " " + str(rhat(listbeta105, 4)))
