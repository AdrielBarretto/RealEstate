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
num_features = ['BEDS','BATHS','SQFT','BUILDING_TYPE_APT',
                'BUILDING_TYPE_COMM','BUILDING_TYPE_CON',
                'BUILDING_TYPE_SFR','GARAGE_Y','POOL_Y','TIME']


encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
num_scaler = StandardScaler()

def get_batches(df, batch_size=500_000):
    n = len(df)
    for i in range(0, n, batch_size):
        batch = df.iloc[i:i+batch_size]
        X_zip = encoder.transform(batch[['ZIP']])  # stays sparse
        X_num = batch[num_features].astype(np.float32)
        X_num_scaled = num_scaler.transform(batch[num_features])
        X_num_sparse = csr_matrix(X_num_scaled)  # convert to sparse for hstack

        # Combine sparse + numeric
        X = hstack([X_zip, X_num_sparse], format='csr')

        # Target
        y = batch['logrent'].values
        yield X, y
model = SGDRegressor(
    loss="squared_error",
    penalty=None,
    max_iter=1,      # 1 epoch per partial_fit call
    learning_rate='invscaling',
    eta0=0.01,
    random_state=24
)
# for X_batch, y_batch in get_batches(cleaned):
#     model.partial_fit(X_batch, y_batch)

train_idx, test_idx = train_test_split(
    cleaned.index, test_size=0.2, random_state=24
)

train_df = cleaned.loc[train_idx]
test_df  = cleaned.loc[test_idx]
encoder.fit(train_df[['ZIP']])
num_scaler.fit(train_df[num_features])
for X_batch, y_batch in get_batches(train_df):
    model.partial_fit(X_batch, y_batch)

y_true = []
y_pred = []

for X_batch, y_batch in get_batches(test_df):
    y_true.append(y_batch)
    y_pred.append(model.predict(X_batch))

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

mse = np.mean((y_true - y_pred)**2)
print("MSE:", mse)

r2 = r2_score(y_true, y_pred)

print("MSE:", mse)
print("R²:", r2)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

sgd = model

zip_feature_names = encoder.get_feature_names_out(['ZIP'])
num_feature_names = num_features

all_features = list(zip_feature_names) + list(num_feature_names)

coef_df = pd.DataFrame({
    "feature": all_features,
    "coef": sgd.coef_
})

print("Intercept:", sgd.intercept_[0])
print(coef_df)
#SGD Large one done
print("Large Linear Done")
