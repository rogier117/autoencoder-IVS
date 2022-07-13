import numpy as np
import pandas as pd
from ipca import InstrumentedPCA
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

df_bal = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')
df_bal['Date'] = pd.to_datetime(df_bal['Date'], format='%Y-%m-%d')
t = np.zeros(df_bal.shape[0]).astype(int)
count = 0
for _ in range(df_bal.shape[0]):
    if _ == 0:
        t[_] = count
    elif df_bal['Date'][_] == df_bal['Date'][_-1]:
        t[_] = count
    else:
        count += 1
        t[_] = count
df_bal['Date'] = t.astype(int)
df_bal = df_bal.drop(columns=["tau_nor", "k_nor", "t"])
covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
covariates = covariates.drop(columns='Date')

gridsize = df_bal[df_bal.Date == df_bal.Date[0]].shape[0]
ID = np.arange(gridsize)
ID = np.tile(ID, len(df_bal.Date.unique()))
df_bal['ID'] = ID
df_bal.insert(0, 'ID', df_bal.pop('ID'))

for col in covariates.columns:
    df_bal[col] = covariates[col].repeat(gridsize).reset_index(drop=True)

cut_off = df_bal.Date.unique()[round(len(df_bal.Date.unique()) * 0.8)]
X_train = df_bal[df_bal.Date <= cut_off]
X_test = df_bal[df_bal.Date > cut_off]

X_train = X_train.sort_values(by=['ID','Date'])
X_train = X_train.set_index(['ID','Date'], drop=True, append=False)
y_train = X_train['IV']
X_train = X_train.drop('IV', axis=1)

X_test = X_test.sort_values(by=['ID','Date'])
X_test = X_test.set_index(['ID','Date'], drop=True, append=False)
y_test = X_test['IV']
X_test = X_test.drop('IV', axis=1)

regr = InstrumentedPCA(n_factors=3, intercept=False, max_iter=200)
regr = regr.fit(X=X_train, y=y_train, data_type='panel')
Gamma, Factors = regr.get_factors(label_ind=True)

alldates = np.array([x[1] for x in X_test.index])
dates = np.unique(alldates)

# n = np.zeros(len(dates)).astype(int)
# for _ in range(len(dates)):
#     n[_] = np.sum(alldates == dates[_])

y_hat = np.zeros(y_test.shape[0])
begin = 0
for date in dates:
    el = (alldates == date).nonzero()[0]
    end = begin + len(el)
    y_hat_temp = regr.predictOOS(X=X_test.iloc[el], y=y_test.iloc[el])
    y_hat_temp = [x[0] for x in y_hat_temp]
    y_hat[begin:end] = y_hat_temp
    begin = end
