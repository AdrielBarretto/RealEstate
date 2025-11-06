import pandas as pd 
import numpy as np 
import sklearn as sc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import tensorflow as tf
import keras
import gc
import matplotlib.pyplot as plt

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
listofzips = cleaned['ZIP'].unique()
mse_list = []
neural_mse_list = []
'''
for each in listofzips:
    cleaned1 = cleaned[cleaned['ZIP'] == each]
    X = cleaned1[['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y','TIME']].fillna(0)
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
    #Neural Network
    x = sc.preprocessing.StandardScaler().fit_transform(X)
    y = (np.array(cleaned1['logrent']) - np.mean(np.array(cleaned1['logrent'])))/np.std(np.array(cleaned1['logrent']))
    nX_train, nX_test, ny_train, ny_test = sc.model_selection.train_test_split(x, y, test_size=0.2)
    hidden_units = 20
    activation = 'sigmoid'
    learning_rate = 0.05
    epochs = 25
    # [5,10,25,50,75,100]
    batch_size = 32
    neural_model = keras.models.Sequential()
    neural_model.add(keras.layers.Dense(input_dim=len(X.columns),
                                 units=hidden_units,
                                 activation=activation))
    # add the output layer
    neural_model.add(keras.layers.Dense(input_dim=hidden_units,
                                 units=1))
    # define our loss function and optimizer
    neural_model.compile(loss='MeanSquaredError',
                  # Adam is a kind of gradient descent
                  optimizer=keras.optimizers.Adam(learning_rate=.01),
                  metrics=['mse'])
    execute = neural_model.fit(nX_train, ny_train, epochs=epochs, batch_size=batch_size)
    test_acc = neural_model.evaluate(nX_test, ny_test)
    neural_mse_list.append(test_acc)
average = np.mean(mse_list)
average2 = np.mean(neural_mse_list)
print(average)
print(average2)
'''
#Straight linear regression
cleaned.dropna(inplace = True)
a = pd.get_dummies(cleaned['ZIP'],drop_first = False, sparse=True)
acols = a.columns
cleaned = cleaned.join(a)
t = ['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y','TIME']+acols.to_list()

X = cleaned[t]
X_train, X_test, y_train, y_test = train_test_split(X, cleaned['logrent'], test_size=0.2,random_state =24)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
y_test = np.array(y_test)
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = 0
for i in range(len(y_test)):
    mse+= (y_test[i]-y_pred[i])**2
mse = mse/len(y_test)

#PCA
del sample, samps, one, new_samps
gc.collect()
a = pd.get_dummies(cleaned['ZIP'],drop_first = False, sparse=True)
ab = a.sparse.to_coo().tocsr()
pca = sc.decomposition.TruncatedSVD(200)
d = pca.fit(ab)
d = pd.DataFrame(d)
pca_set = cleaned.join(d)
colpca = ['BEDS','BATHS','SQFT', 'BUILDING_TYPE_APT','BUILDING_TYPE_COMM', 'BUILDING_TYPE_CON','BUILDING_TYPE_SFR', 'GARAGE_Y', 'POOL_Y','TIME']+d.columns.to_list()
Xpca = pca_set[colpca]
X_trainpca, X_testpca, y_trainpca, y_testpca = train_test_split(Xpca, pca_set['logrent'], test_size=0.2, random_state =24)
X_trainpca.columns = X_train.columns.astype(str)
X_testpca.columns = X_test.columns.astype(str)
y_testpca = np.array(y_testpca)
model = LinearRegression()
model.fit(X_trainpca, y_trainpca)
y_predpca = model.predict(X_testpca)
msepca = 0
for i in range(len(y_testpca)):
    msepca += (y_testpca[i]-y_predpca[i])**2
msepca = msepca/len(y_testpca)

#Neural net 
x = sc.preprocessing.StandardScaler().fit_transform(X)
y = (np.array(pca_set['logrent']) - np.mean(np.array(pca_set['logrent'])))/np.std(np.array(pca_set['logrent']))
nX_train, nX_test, ny_train, ny_test = sc.model_selection.train_test_split(x, y, test_size=0.2, random_state =24)
hidden_units = 20
activation = 'sigmoid'
learning_rate = 0.05
epochs = 25
    # [5,10,25,50,75,100]
batch_size = 32
neural_model = keras.models.Sequential()
neural_model.add(keras.layers.Dense(input_dim=len(X.columns),
                                 units=hidden_units,
                                 activation=activation))
    # add the output layer
neural_model.add(keras.layers.Dense(input_dim=hidden_units,
                                 units=1))
# define our loss function and optimizer
neural_model.compile(loss='MeanSquaredError',
                  # Adam is a kind of gradient descent
    optimizer=keras.optimizers.Adam(learning_rate=.01),
    metrics=['mse'])
execute = neural_model.fit(nX_train, ny_train, epochs=epochs, batch_size=batch_size)
test_acc = neural_model.evaluate(nX_test, ny_test)




#filter by number of entries
#log it 
#0-1 columns for neural network 
#MCA 
#Straight eregression 
#Bayesian Hierarchical