import numpy as np
import pandas as pd
from ipca import InstrumentedPCA
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm


def balanced_preprocessing(df_bal_f, covariates_f, split=0.8):
    df_bal_f['Date'] = df_bal_f.t.astype(int)
    df_bal_f = df_bal_f.drop(columns=["tau_nor", "k_nor", "t"])

    gridsize = df_bal_f[df_bal_f.Date == df_bal_f.Date[0]].shape[0]
    ID = np.arange(gridsize)
    ID = np.tile(ID, len(df_bal_f.Date.unique()))
    df_bal_f['ID'] = ID
    df_bal_f.insert(0, 'ID', df_bal_f.pop('ID'))

    covariates_f = covariates_f.iloc[21:]
    for col in covariates_f.columns:
        df_bal_f[col] = covariates_f[col].repeat(gridsize).reset_index(drop=True)

    # ADD COVARIATES WHICH MODEL THE NONLINEARITIES WITHIN THE MONEYNESS: (moneyness - 1)^2, moneyness * tau(, tau^2)
    df_bal_f['moneynessdev'] = (df_bal_f.moneyness - 1) ** 2
    df_bal_f['crossterm'] = df_bal_f.moneyness * df_bal_f.daystoex
    df_bal_f['daystoexsq'] = df_bal_f.daystoex ** 2

    cut_off = df_bal_f.Date.unique()[round(len(df_bal_f.Date.unique()) * split)]
    X_train_f = df_bal_f[df_bal_f.Date <= cut_off]
    X_test_f = df_bal_f[df_bal_f.Date > cut_off]

    X_train_f = X_train_f.sort_values(by=['ID', 'Date'])
    X_train_f = X_train_f.set_index(['ID', 'Date'], drop=True, append=False)
    y_train_f = X_train_f['IV']
    X_train_f = X_train_f.drop('IV', axis=1)

    X_test_f = X_test_f.sort_values(by=['ID', 'Date'])
    X_test_f = X_test_f.set_index(['ID', 'Date'], drop=True, append=False)
    y_test_f = X_test_f['IV']
    X_test_f = X_test_f.drop('IV', axis=1)

    return X_train_f, y_train_f, X_test_f, y_test_f


def ipca_train(X_train_f, y_train_f, n_factors=3, max_iter=200):
    model = InstrumentedPCA(n_factors=n_factors, intercept=False, max_iter=max_iter)
    model = model.fit(X=X_train_f, y=y_train_f, data_type='panel')
    return model


def ipca_test(X_test_f, y_test_f, model):
    alldates = np.array([x[1] for x in X_test_f.index])
    dates = np.unique(alldates)

    y_hat = np.zeros(y_test_f.shape[0])
    y_hat[:] = np.NaN
    for date in dates:
        el = (alldates == date).nonzero()[0]
        y_hat_temp = model.predictOOS(X=X_test_f.iloc[el], y=y_test_f.iloc[el])
        y_hat_temp = [x[0] for x in y_hat_temp]
        y_hat[el] = y_hat_temp
    return y_hat


# Use unbalanced data for training

def unbalanced_preprocessing(df_unb, covariates_f, split=0.8):
    df_unb['Date'] = df_unb.t.astype(int)
    df_unb = df_unb.drop(columns=["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", "t",
                                  "price", "moneynessdev", "q"])

    ID = np.arange(start=0, stop=df_unb.shape[0])
    df_unb['ID'] = ID
    df_unb.insert(0, 'ID', df_unb.pop('ID'))

    covariates_f = covariates_f.iloc[21:]
    temp = np.zeros((df_unb.shape[0], covariates_f.shape[1]))
    temp2 = covariates_f.values
    date = df_unb.Date.values
    for _ in range(len(temp)):
        temp[_, :] = temp2[date[_], :]

    df_unb[covariates_f.columns] = temp

    # ADD COVARIATES WHICH MODEL THE NONLINEARITIES WITHIN THE MONEYNESS: (moneyness - 1)^2, moneyness * tau(, tau^2)
    df_unb['moneynessdev'] = (df_unb.moneyness - 1) ** 2
    df_unb['crossterm'] = df_unb.moneyness * df_unb.daystoex
    df_unb['daystoexsq'] = df_unb.daystoex ** 2

    cut_off = df_unb.Date.unique()[round(len(df_unb.Date.unique()) * split)]
    X_train_f = df_unb[df_unb.Date <= cut_off]
    X_test_f = df_unb[df_unb.Date > cut_off]

    X_train_f = X_train_f.sort_values(by=['ID', 'Date'])
    X_train_f = X_train_f.set_index(['ID', 'Date'], drop=True, append=False)
    y_train_f = X_train_f['IV']
    X_train_f = X_train_f.drop('IV', axis=1)

    X_test_f = X_test_f.sort_values(by=['ID', 'Date'])
    X_test_f = X_test_f.set_index(['ID', 'Date'], drop=True, append=False)
    y_test_f = X_test_f['IV']
    X_test_f = X_test_f.drop('IV', axis=1)

    return X_train_f, y_train_f, X_test_f, y_test_f


def forecast_preprocessing(X_train_in, y_train_in, X_test_in, y_test_in, covariates_f, horizon=1):
    covariates_f = covariates_f.iloc[21 - horizon:-horizon, :]
    temp2 = covariates_f.values

    tr_temp = np.zeros((X_train_in.shape[0], covariates_f.shape[1]))
    te_temp = np.zeros((X_test_in.shape[0], covariates_f.shape[1]))
    tr_date = np.array(X_train_in.index.get_level_values('Date'))
    te_date = np.array(X_test_in.index.get_level_values('Date'))

    for _ in range(len(X_train_in)):
        tr_temp[_, :] = temp2[tr_date[_], :]

    for _ in range(len(X_test_in)):
        te_temp[_, :] = temp2[te_date[_], :]

    X_train_f = X_train_in.copy()
    X_train_f.iloc[:, 2:2 + covariates_f.shape[1]] = tr_temp

    X_test_f = X_test_in[:].copy()
    X_test_f.iloc[:, 2:2 + covariates_f.shape[1]] = te_temp

    y_train_f = y_train_in
    y_test_f = y_test_in

    # last_date = np.max(X_train_in.index.get_level_values('Date'))
    # last_date_test = np.max(X_test_in.index.get_level_values('Date'))
    #
    # X_train = X_train_in[X_train_in.index.get_level_values('Date') <= last_date - horizon]
    # y_train = y_train_in[y_train_in.index.get_level_values('Date') >= horizon]
    #
    # y_train.index = y_train.index.set_levels(y_train.index.levels[1] - horizon, level=1)
    #
    # X_test1 = X_train_in[X_train_in.index.get_level_values('Date') > last_date - horizon]
    # X_test2 = X_test_in[X_test_in.index.get_level_values('Date') <= last_date_test - horizon]
    # X_test = pd.concat([X_test1, X_test2])
    # X_test = X_test.sort_index(axis=0)
    #
    # y_test = y_test_in
    # y_test.index = y_test.index.set_levels(y_test.index.levels[1] - horizon, level=1)

    return X_train_f, y_train_f, X_test_f, y_test_f

# balanced

df_bal = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')
df_bal['Date'] = pd.to_datetime(df_bal['Date'], format='%Y-%m-%d')
covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
covariates = covariates.drop(columns='Date')

X_train, y_train, X_test, y_test = balanced_preprocessing(df_bal_f=df_bal, covariates_f=covariates, split=0.8)
# model = ipca_train(X_train=X_train, y_train=y_train, n_factors=3, max_iter=10)
# y_hat = ipca_test(X_test=X_test, y_test=y_test, model=model)
# Gamma, Factors = model.get_factors(label_ind=True)

# # unbalanced
#
# df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
# df_unb['date'] = pd.to_datetime(df_unb['date'], format='%Y-%m-%d')
#
# covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
# covariates = covariates.drop(columns='Date')
#
# X_train, y_train, X_test, y_test = unbalanced_preprocessing(df_unb, covariates, split=0.8)
# model = ipca_train(X_train=X_train, y_train=y_train, n_factors=3, max_iter=10)
# y_hat_unb = ipca_test(X_test=X_test, y_test=y_test, model=model)
# Gamma, Factors = model.get_factors(label_ind=True)

# # Forecast

X_trainf, y_trainf, X_testf, y_testf = forecast_preprocessing(X_train_in=X_train, y_train_in=y_train, X_test_in=X_test,
                                                              y_test_in=y_test, covariates_f=covariates, horizon=1)
# modelf = ipca_train(X_train=X_trainf, y_train=y_trainf, n_factors=3, max_iter=10)
# y_hatf = ipca_test(X_test=X_testf, y_test=y_testf, model=modelf)
# Gamma, Factors = modelf.get_factors(label_ind=True)
