import numpy as np
import pandas as pd
from ipca import InstrumentedPCA
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm


def balanced_preprocessing(df_bal, covariates, split=0.8):
    df_bal['Date'] = df_bal.t.astype(int)
    df_bal = df_bal.drop(columns=["tau_nor", "k_nor", "t"])

    gridsize = df_bal[df_bal.Date == df_bal.Date[0]].shape[0]
    ID = np.arange(gridsize)
    ID = np.tile(ID, len(df_bal.Date.unique()))
    df_bal['ID'] = ID
    df_bal.insert(0, 'ID', df_bal.pop('ID'))

    covariates = covariates.iloc[21:]
    for col in covariates.columns:
        df_bal[col] = covariates[col].repeat(gridsize).reset_index(drop=True)

    # ADD COVARIATES WHICH MODEL THE NONLINEARITIES WITHIN THE MONEYNESS: (moneyness - 1)^2, moneyness * tau(, tau^2)
    df_bal['moneynessdev'] = (df_bal.moneyness - 1) ** 2
    df_bal['crossterm'] = df_bal.moneyness * df_bal.daystoex
    df_bal['daystoexsq'] = df_bal.daystoex ** 2

    cut_off = df_bal.Date.unique()[round(len(df_bal.Date.unique()) * split)]
    X_train = df_bal[df_bal.Date <= cut_off]
    X_test = df_bal[df_bal.Date > cut_off]

    X_train = X_train.sort_values(by=['ID', 'Date'])
    X_train = X_train.set_index(['ID', 'Date'], drop=True, append=False)
    y_train = X_train['IV']
    X_train = X_train.drop('IV', axis=1)

    X_test = X_test.sort_values(by=['ID', 'Date'])
    X_test = X_test.set_index(['ID', 'Date'], drop=True, append=False)
    y_test = X_test['IV']
    X_test = X_test.drop('IV', axis=1)

    return X_train, y_train, X_test, y_test


def ipca_train(X_train, y_train, n_factors=3, max_iter=200):
    model = InstrumentedPCA(n_factors=n_factors, intercept=False, max_iter=max_iter)
    model = model.fit(X=X_train, y=y_train, data_type='panel')
    return model


def ipca_test(X_test, y_test, model):
    alldates = np.array([x[1] for x in X_test.index])
    dates = np.unique(alldates)

    y_hat = np.zeros(y_test.shape[0])
    y_hat[:] = np.NaN
    for date in dates:
        el = (alldates == date).nonzero()[0]
        y_hat_temp = model.predictOOS(X=X_test.iloc[el], y=y_test.iloc[el])
        y_hat_temp = [x[0] for x in y_hat_temp]
        y_hat[el] = y_hat_temp
    return y_hat

# df_bal = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')
# df_bal['Date'] = pd.to_datetime(df_bal['Date'], format='%Y-%m-%d')
# covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
# covariates = covariates.drop(columns='Date')
#
# X_train, y_train, X_test, y_test = balanced_preprocessing(df_bal=df_bal, covariates=covariates, split=0.8)
# model = ipca_train(X_train=X_train, y_train=y_train, n_factors=3, max_iter=100)
# y_hat = ipca_test(X_test=X_test, y_test=y_test, model=model)
# Gamma, Factors = model.get_factors(label_ind=True)


# Use unbalanced data for training

def unbalanced_preprocessing(df_unb, covariates, split=0.8):
    df_unb['Date'] = df_unb.t.astype(int)
    df_unb = df_unb.drop(columns=["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", "t",
                                  "price", "moneynessdev", "q"])

    ID = np.arange(start=0, stop=df_unb.shape[0])
    df_unb['ID'] = ID
    df_unb.insert(0, 'ID', df_unb.pop('ID'))

    covariates = covariates.iloc[21:]
    temp = np.zeros((df_unb.shape[0], covariates.shape[1]))
    temp2 = covariates.values
    date = df_unb.Date.values
    for _ in range(len(temp)):
        temp[_, :] = temp2[date[_],:]

    df_unb[covariates.columns] = temp

    # ADD COVARIATES WHICH MODEL THE NONLINEARITIES WITHIN THE MONEYNESS: (moneyness - 1)^2, moneyness * tau(, tau^2)
    df_unb['moneynessdev'] = (df_unb.moneyness - 1) ** 2
    df_unb['crossterm'] = df_unb.moneyness * df_unb.daystoex
    df_unb['daystoexsq'] = df_unb.daystoex ** 2

    cut_off = df_unb.Date.unique()[round(len(df_unb.Date.unique()) * split)]
    X_train = df_unb[df_unb.Date <= cut_off]
    X_test = df_unb[df_unb.Date > cut_off]

    X_train = X_train.sort_values(by=['ID', 'Date'])
    X_train = X_train.set_index(['ID', 'Date'], drop=True, append=False)
    y_train = X_train['IV']
    X_train = X_train.drop('IV', axis=1)

    X_test = X_test.sort_values(by=['ID', 'Date'])
    X_test = X_test.set_index(['ID', 'Date'], drop=True, append=False)
    y_test = X_test['IV']
    X_test = X_test.drop('IV', axis=1)

    return X_train, y_train, X_test, y_test


df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
df_unb['date'] = pd.to_datetime(df_unb['date'], format='%Y-%m-%d')

covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
covariates = covariates.drop(columns='Date')

X_train, y_train, X_test, y_test = unbalanced_preprocessing(df_unb, covariates, split=0.8)
model = ipca_train(X_train=X_train, y_train=y_train, n_factors=3, max_iter=10)
y_hat_unb = ipca_test(X_test=X_test, y_test=y_test, model=model)
Gamma, Factors = model.get_factors(label_ind=True)
