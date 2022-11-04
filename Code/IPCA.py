import numpy as np
import pandas as pd
from ipca import InstrumentedPCA
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from keras import layers
from Code.performance_measure import rsq, rsq_recession, ivrmse
from Code.VAR_functions import train_VAR_model, test_VAR_model
import tensorflow as tf

import pickle
from statsmodels.tsa.stattools import adfuller


def balanced_preprocessing(df_bal_f, covariates_f, split=0.8):
    # change date to integer from 0 to T-1
    df_bal_f = df_bal_f.copy()
    df_bal_f['Date'] = df_bal_f.t.astype(int)
    df_bal_f = df_bal_f.drop(columns=["tau_nor", "k_nor", "t"])

    # ID shows which tau-k pair it is
    gridsize = df_bal_f[df_bal_f.Date == df_bal_f.Date[0]].shape[0]
    ID = np.arange(gridsize)
    ID = np.tile(ID, len(df_bal_f.Date.unique()))
    df_bal_f['ID'] = ID
    df_bal_f.insert(0, 'ID', df_bal_f.pop('ID'))

    # covariates start 21 days before start sample
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

    sc_x = StandardScaler()

    X_train_f_val = sc_x.fit_transform(X_train_f.values)
    X_train_f = pd.DataFrame(X_train_f_val, index=X_train_f.index, columns=X_train_f.columns)

    X_test_f_val = sc_x.transform(X_test_f.values)
    X_test_f = pd.DataFrame(X_test_f_val, index=X_test_f.index, columns=X_test_f.columns)

    sc_y = StandardScaler()
    y_train_f_val = sc_y.fit_transform(y_train_f.values.reshape(-1, 1)).reshape(-1, )
    y_train_f = pd.Series(y_train_f_val, index=y_train_f.index)

    y_test_f_val = sc_y.transform(y_test_f.values.reshape(-1, 1)).reshape(-1, )
    y_test_f = pd.Series(y_test_f_val, index=y_test_f.index)

    return X_train_f, y_train_f, X_test_f, y_test_f, sc_x, sc_y


def ipca_train(X_train_f, y_train_f, n_factors=3, max_iter=200):
    model = InstrumentedPCA(n_factors=n_factors, intercept=False, max_iter=max_iter)
    model = model.fit(X=X_train_f, y=y_train_f, data_type='panel')
    return model


def ipca_test(X_test_f, y_test_f, model, n_factors):
    alldates = np.array([x[1] for x in X_test_f.index])
    dates = np.unique(alldates)

    y_hat = np.zeros(y_test_f.shape[0])
    y_hat[:] = np.NaN

    factors_hat = np.zeros((dates.shape[0], n_factors))
    factors_hat[:, :] = np.NaN
    _ = 0
    for date in dates:
        el = (alldates == date).nonzero()[0]
        y_hat_temp, factors_hat_temp = model.predictOOS(X=X_test_f.iloc[el], y=y_test_f.iloc[el])
        y_hat_temp = [x[0] for x in y_hat_temp]
        y_hat[el] = y_hat_temp
        factors_hat[_, :] = np.transpose(factors_hat_temp)
        _ += 1
    return y_hat, factors_hat


# Use unbalanced data for training

def unbalanced_preprocessing(df_unb, covariates_f, sc_x, sc_y, split=0.8):
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

    # normalize everything according to the same scales as for the balanced case
    X_train_f_val = sc_x.transform(X_train_f.values)
    X_train_f = pd.DataFrame(X_train_f_val, index=X_train_f.index, columns=X_train_f.columns)

    X_test_f_val = sc_x.transform(X_test_f.values)
    X_test_f = pd.DataFrame(X_test_f_val, index=X_test_f.index, columns=X_test_f.columns)

    y_train_f_val = sc_y.transform(y_train_f.values.reshape(-1, 1)).reshape(-1, )
    y_train_f = pd.Series(y_train_f_val, index=y_train_f.index)

    y_test_f_val = sc_y.transform(y_test_f.values.reshape(-1, 1)).reshape(-1, )
    y_test_f = pd.Series(y_test_f_val, index=y_test_f.index)

    return X_train_f, y_train_f, X_test_f, y_test_f


def forecast_preprocessing(X_train_in, y_train_in, X_test_in, y_test_in, covariates_f, sc_x, sc_y, horizon=1,
                           balanced=True):
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

    # if the balanced dataset is given, then the standardscaler should be trained over, if not, use the one trained for balanced
    # Standardscalers are already used for the 'regular' preprocessing (I think)
    # if balanced:
    #     sc_x = StandardScaler()
    #
    #     X_train_f_val = sc_x.fit_transform(X_train_f.values)
    #     X_train_f = pd.DataFrame(X_train_f_val, index=X_train_f.index, columns=X_train_f.columns)
    # #
    # #     sc_y = StandardScaler()
    # #     y_train_f_val = sc_y.fit_transform(y_train_f.values.reshape(-1, 1)).reshape(-1, )
    # #     y_train_f = pd.Series(y_train_f_val, index=y_train_f.index)
    # #
    # else:
    #     X_train_f_val = sc_x.transform(X_train_f.values)
    #     X_train_f = pd.DataFrame(X_train_f_val, index=X_train_f.index, columns=X_train_f.columns)
    #
    #     y_train_f_val = sc_y.transform(y_train_f.values.reshape(-1, 1)).reshape(-1, )
    #     y_train_f = pd.Series(y_train_f_val, index=y_train_f.index)

    # test should be trained the same independent of balanced/unbalanced
    X_train_f_val = sc_x.transform(X_train_f.values)
    X_train_f = pd.DataFrame(X_train_f_val, index=X_train_f.index, columns=X_train_f.columns)
    X_test_f_val = sc_x.transform(X_test_f.values)
    X_test_f = pd.DataFrame(X_test_f_val, index=X_test_f.index, columns=X_test_f.columns)
    #
    # y_test_f_val = sc_y.transform(y_test_f.values.reshape(-1, 1)).reshape(-1, )
    # y_test_f = pd.Series(y_test_f_val, index=y_test_f.index)

    return X_train_f, y_train_f, X_test_f, y_test_f


def nn_preprocessing(covariates_f, factors_f, factors_test_f, horizon=1):
    factors_f = factors_f.values
    factors_f = np.transpose(factors_f)

    X_f = covariates_f.iloc[21:].values
    X_f = np.append(X_f, np.append(factors_f, factors_test_f, axis=0), axis=1)
    X_train_f = X_f[:factors_f.shape[0] - horizon, :]
    X_test_f = X_f[factors_f.shape[0] - horizon:-horizon, :]
    y_train_f = factors_f[horizon:, :]
    y_test_f = factors_test_f[:, :]

    return X_train_f, y_train_f, X_test_f, y_test_f


def forecast_train(X_f, y_f, n_epochs=100, batch_size=64, width=64):
    # Set seed
    tf.random.set_seed(1234)

    n_factors = y_f.shape[1]
    n_inputs = X_f.shape[1]

    input = keras.Input(shape=(n_inputs,))
    layer_1 = layers.Dense(width, activation='relu')(input)
    layer_15 = layers.Dense(width, activation='relu')(layer_1)
    layer_2 = layers.Dense(n_factors, activation='linear')(layer_15)
    model = keras.Model(inputs=input, outputs=layer_2)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_f, y_f,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=True)
    return model


def forecast_test(model, X_test_nn_f, X_test_f, gamma, ar=False, factors=None):
    X_test_fv = X_test_f.values
    dates = np.array(X_test_f.index.get_level_values('Date'))
    mindate = np.min(dates)
    gamma = gamma.values
    factors_hat = model.predict(X_test_nn_f)
    if ar:
        factors_hat = X_test_nn_f[:, -factors:]

    y_hat_nn = np.zeros(X_test_f.shape[0])
    for _ in range(y_hat_nn.shape[0]):
        # (c * gamma) * f
        y_hat_nn[_] = np.dot(np.matmul(X_test_fv[_, :], gamma), factors_hat[dates[_] - mindate])

    return y_hat_nn


def forecast_test_VAR(gamma, f_hat, X_test_f):
    X_test_fv = X_test_f.values
    dates = np.array(X_test_f.index.get_level_values('Date'))
    mindate = np.min(dates)
    gamma = gamma.values

    y_hat_nn = np.zeros(X_test_f.shape[0])
    for _ in range(y_hat_nn.shape[0]):
        # (c * gamma) * f
        y_hat_nn[_] = np.dot(np.matmul(X_test_fv[_, :], gamma), f_hat[dates[_] - mindate])

    return y_hat_nn


def bootstrap_data(bs, X_train, y_train):
    dates = np.array([x[1] for x in X_train.index])
    maxdate = np.max(dates) + 1

    X_train_out = list()
    X_test_out = list()

    y_train_out = list()
    y_test_out = list()

    for _ in range(bs):
        cut_begin = round((maxdate * _)/bs)
        cut_end = round((maxdate * (_+1))/bs)
        sel1 = dates >= cut_begin
        sel2 = dates < cut_end
        # Selection of elements for test
        el = sel1 * sel2
        X_test_out.append(X_train[el])
        y_test_out.append(y_train[el])
        # Selection of elements for train
        elc = np.array([not elem for elem in el])
        X_train_out.append(X_train[elc])
        y_train_out.append(y_train[elc])

    return X_train_out, X_test_out, y_train_out, y_test_out


def bootstrap(X_train, bs, y_train, sc_y):
    done = False
    removed = list()

    X_train_f = X_train.copy()
    y_train_f = y_train.copy()

    X_train_temp, X_test_temp, y_train_temp, y_test_temp = bootstrap_data(bs=bs, X_train=X_train_f, y_train=y_train_f)
    rmse_temp = np.zeros(bs)

    for i in range(bs):
        model = ipca_train(X_train_f=X_train_temp[i], y_train_f=y_train_temp[i], n_factors=3, max_iter=50)
        y_hat, f_hat = ipca_test(X_test_f=X_test_temp[i], y_test_f=y_test_temp[i], model=model, n_factors=3)
        rmse_temp[i] = ivrmse(y_test=y_test_temp[i], y_hat=y_hat, sc_y=sc_y)

    boundary = np.mean(rmse_temp)

    while not done:
        rmse_unb = np.zeros(X_train_f.shape[1])
        for _ in range(X_train_f.shape[1]):
            X_train_col = X_train_f.drop(X_train_f.columns[_], axis=1)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = bootstrap_data(bs=bs, X_train=X_train_col,
                                                                                  y_train=y_train)
            rmse_temp = np.zeros(bs)

            for i in range(bs):
                model = ipca_train(X_train_f=X_train_temp[i], y_train_f=y_train_temp[i], n_factors=3, max_iter=50)
                y_hat, f_hat = ipca_test(X_test_f=X_test_temp[i], y_test_f=y_test_temp[i], model=model, n_factors=3)
                rmse_temp[i] = ivrmse(y_test=y_test_temp[i], y_hat=y_hat, sc_y=sc_y)

            rmse_unb[_] = np.mean(rmse_temp)

        if np.max(rmse_unb) < boundary:
            done = True
        else:
            arg = np.argmax(rmse_unb)
            removed.append(X_train_f.columns[arg])
            X_train_f = X_train_f.drop(X_train.columns[arg], axis=1)
            boundary = np.max(rmse_unb)
    return removed


# balanced

df_bal = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')
df_bal['Date'] = pd.to_datetime(df_bal['Date'], format='%Y-%m-%d')
df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
df_unb['date'] = pd.to_datetime(df_unb['date'], format='%Y-%m-%d')
covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
covariates = covariates.drop(columns='Date')

# Variable selection
# rem_var = bootstrap(X_train=X_train, y_train=y_train, bs=5, sc_y=sc_y)
rem_var = ['SPXM', 'FFER', 'US10YMY', 'GDPBBK']
covariates = covariates.drop(columns=rem_var)

X_train, y_train, X_test, y_test, sc_x, sc_y = balanced_preprocessing(df_bal_f=df_bal, covariates_f=covariates,
                                                                      split=0.8)
X_train_unb, y_train_unb, X_test_unb, y_test_unb = unbalanced_preprocessing(df_unb, covariates, sc_x=sc_x, sc_y=sc_y, split=0.8)



# X_train = X_train.drop(columns=rem_var)
# X_test = X_test.drop(columns=rem_var)
# X_train_unb = X_train_unb.drop(columns=rem_var)
# X_test_unb = X_test_unb.drop(columns=rem_var)


# # Train models for 1 to 6 factors
#
# rmse = np.zeros(6)
# rmse_unb = np.zeros(6)
#
# for _ in range(6):
#     # model = ipca_train(X_train_f=X_train_unb, y_train_f=y_train_unb, n_factors=_+1, max_iter=300)
#     tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\IPCA\IPCAu_" + str(_+1) + "f_0h"
#     model = pickle.load(open(tempdir, "rb"))
#     y_hat, f_hat = ipca_test(X_test_f=X_test, y_test_f=y_test, model=model, n_factors=_+1)
#     rmse[_] = ivrmse(y_test=y_test, y_hat=y_hat, sc_y=sc_y)
#
#     y_hat_unb, f_hat_unb = ipca_test(X_test_f=X_test_unb, y_test_f=y_test_unb, model=model, n_factors=_+1)
#     rmse_unb[_] = ivrmse(y_test=y_test_unb, y_hat=y_hat_unb, sc_y=sc_y)
#
#     # tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\IPCA\IPCAu_" + str(_ + 1) + "f_0h"
#     # pickle.dump(model, open(tempdir, 'wb'))
#
#
# # Factor interpretation 1-factor model
# tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\IPCA\IPCAb_1f_0h"
# trained_model = pickle.load(open(tempdir, "rb"))
# factors_bal_train = np.array(trained_model.get_factors(label_ind=True)[1]).transpose()
# # factors_bal_train = ipca_test(X_test_f=X_train, y_test_f=y_train, model=trained_model, n_factors=3)[1]
# factors_bal_test = ipca_test(X_test_f=X_test, y_test_f=y_test, model=trained_model, n_factors=1)[1]
# factors = np.concatenate((factors_bal_train, factors_bal_test), axis=0)



# Forecast
rmse = np.zeros((3, 6))
rmse_u = np.zeros((3, 6))

rmse_rec = np.zeros((3, 6))
rmse_ar = np.zeros((3, 6))

horizon = np.array([1, 5, 21])

# for _ in range(6):
#     for h in range(3):
#         X_trainf, y_trainf, X_testf, y_testf = forecast_preprocessing(X_train_in=X_train, y_train_in=y_train,
#                                                                       X_test_in=X_test, y_test_in=y_test,
#                                                                       covariates_f=covariates, sc_x=sc_x, sc_y=sc_y,
#                                                                       horizon=horizon[h], balanced=True)
#
#         X_trainf_unb, y_trainf_unb, X_testf_unb, y_testf_unb = forecast_preprocessing(X_train_in=X_train_unb, y_train_in=y_train_unb,
#                                                                                       X_test_in=X_test_unb, y_test_in=y_test_unb,
#                                                                                       covariates_f=covariates, sc_x=sc_x, sc_y=sc_y,
#                                                                                       horizon=horizon[h], balanced=False)
#
#         modelf = ipca_train(X_train_f=X_trainf_unb, y_train_f=y_trainf_unb, n_factors=_+1, max_iter=75)
#         y_hatf, factors_hatf = ipca_test(X_test_f=X_testf_unb, y_test_f=y_testf_unb, model=modelf, n_factors=_+1)
#         gammaf, factorsf = modelf.get_factors(label_ind=True)
#         X_train_nn, y_train_nn, X_test_nn, y_test_nn = nn_preprocessing(covariates_f=covariates, factors_f=factorsf,
#                                                                         factors_test_f=factors_hatf, horizon=horizon[h])
#         model_nn = forecast_train(X_f=X_train_nn, y_f=y_train_nn, n_epochs=100, batch_size=84, width=64)
#         y_hatff = forecast_test(model=model_nn, X_test_nn_f=X_test_nn, X_test_f=X_testf_unb, gamma=gammaf)
#         rmse[h, _] = ivrmse(y_test=y_testf_unb, y_hat=y_hatff, sc_y=sc_y)
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAu_" + str(_ + 1) + "f_" + str(horizon[h]) + "h1"
#         pickle.dump(modelf, open(tempdir, 'wb'))
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAu_" + str(_ + 1) + "f_" + str(horizon[h]) + "h0"
#         model_nn.save(tempdir)


# for h in range(3):
#     X_trainf, y_trainf, X_testf, y_testf = forecast_preprocessing(X_train_in=X_train, y_train_in=y_train,
#                                                                   X_test_in=X_test, y_test_in=y_test,
#                                                                   covariates_f=covariates, sc_x=sc_x, sc_y=sc_y,
#                                                                   horizon=horizon[h], balanced=True)
#
#     X_trainf_unb, y_trainf_unb, X_testf_unb, y_testf_unb = forecast_preprocessing(X_train_in=X_train_unb,
#                                                                                   y_train_in=y_train_unb,
#                                                                                   X_test_in=X_test_unb,
#                                                                                   y_test_in=y_test_unb,
#                                                                                   covariates_f=covariates, sc_x=sc_x,
#                                                                                   sc_y=sc_y,
#                                                                                   horizon=horizon[h], balanced=False)
#     for _ in range(6):
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAu_" + str(_ + 1) + "f_" + str(horizon[h]) + "h1"
#         trained_model = pickle.load(open(tempdir, "rb"))
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAu_" + str(_ + 1) + "f_" + str(horizon[h]) + "h0"
#         trained_model_nn = keras.models.load_model(tempdir)
#
#         y_hatf, factors_hatf = ipca_test(X_test_f=X_testf_unb, y_test_f=y_testf_unb, model=trained_model, n_factors=_ + 1) # Change between b and u
#         gammaf, factorsf = trained_model.get_factors(label_ind=True)
#         X_train_nn, y_train_nn, X_test_nn, y_test_nn = nn_preprocessing(covariates_f=covariates, factors_f=factorsf,
#                                                                         factors_test_f=factors_hatf, horizon=horizon[h])
#         y_hatff = forecast_test(model=trained_model_nn, X_test_nn_f=X_test_nn, X_test_f=X_testf, gamma=gammaf)
#         rmse[h, _] = ivrmse(y_test=y_testf, y_hat=y_hatff, sc_y=sc_y)
#
#         y_hatff_unb = forecast_test(model=trained_model_nn, X_test_nn_f=X_test_nn, X_test_f=X_testf_unb, gamma=gammaf)
#         rmse_u[h, _] = ivrmse(y_test=y_testf_unb, y_hat=y_hatff_unb, sc_y=sc_y)


# Forecast test without recession
# for h in range(3):
#     X_trainf, y_trainf, X_testf, y_testf = forecast_preprocessing(X_train_in=X_train, y_train_in=y_train,
#                                                                   X_test_in=X_test, y_test_in=y_test,
#                                                                   covariates_f=covariates, sc_x=sc_x, sc_y=sc_y,
#                                                                   horizon=horizon[h], balanced=True)
#
#     X_trainf_unb, y_trainf_unb, X_testf_unb, y_testf_unb = forecast_preprocessing(X_train_in=X_train_unb,
#                                                                                   y_train_in=y_train_unb,
#                                                                                   X_test_in=X_test_unb,
#                                                                                   y_test_in=y_test_unb,
#                                                                                   covariates_f=covariates, sc_x=sc_x,
#                                                                                   sc_y=sc_y,
#                                                                                   horizon=horizon[h], balanced=False)
#     for _ in range(6):
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAb_" + str(_ + 1) + "f_" + str(horizon[h]) + "h1"
#         trained_model = pickle.load(open(tempdir, "rb"))
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAb_" + str(_ + 1) + "f_" + str(horizon[h]) + "h0"
#         trained_model_nn = keras.models.load_model(tempdir)
#
#         y_hatf, factors_hatf = ipca_test(X_test_f=X_testf_unb, y_test_f=y_testf_unb, model=trained_model, n_factors=_ + 1)
#         gammaf, factorsf = trained_model.get_factors(label_ind=True)
#         X_train_nn, y_train_nn, X_test_nn, y_test_nn = nn_preprocessing(covariates_f=covariates, factors_f=factorsf,
#                                                                         factors_test_f=factors_hatf, horizon=horizon[h])
#         X_test_nn = X_test_nn[:522, :]
#         alldates = np.array([x[1] for x in X_testf_unb.index]) - 4030
#         X_testf_unb1 = X_testf_unb[alldates < 522]
#         y_hatff_unb = forecast_test(model=trained_model_nn, X_test_nn_f=X_test_nn, X_test_f=X_testf_unb1, gamma=gammaf)
#         rmse_rec[h, _] = ivrmse(y_test=y_testf_unb.iloc[:y_hatff_unb.shape[0]], y_hat=y_hatff_unb, sc_y=sc_y)



# # Forecast test with ar term
# for h in range(3):
#     X_trainf, y_trainf, X_testf, y_testf = forecast_preprocessing(X_train_in=X_train, y_train_in=y_train,
#                                                                   X_test_in=X_test, y_test_in=y_test,
#                                                                   covariates_f=covariates, sc_x=sc_x, sc_y=sc_y,
#                                                                   horizon=horizon[h], balanced=True)
#
#     X_trainf_unb, y_trainf_unb, X_testf_unb, y_testf_unb = forecast_preprocessing(X_train_in=X_train_unb,
#                                                                                   y_train_in=y_train_unb,
#                                                                                   X_test_in=X_test_unb,
#                                                                                   y_test_in=y_test_unb,
#                                                                                   covariates_f=covariates, sc_x=sc_x,
#                                                                                   sc_y=sc_y,
#                                                                                   horizon=horizon[h], balanced=False)
#     for _ in range(6):
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAb_" + str(_ + 1) + "f_" + str(horizon[h]) + "h1"
#         trained_model = pickle.load(open(tempdir, "rb"))
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAb_" + str(_ + 1) + "f_" + str(horizon[h]) + "h0"
#         trained_model_nn = keras.models.load_model(tempdir)
#
#         y_hatf, factors_hatf = ipca_test(X_test_f=X_testf_unb, y_test_f=y_testf_unb, model=trained_model, n_factors=_ + 1)
#         gammaf, factorsf = trained_model.get_factors(label_ind=True)
#         X_train_nn, y_train_nn, X_test_nn, y_test_nn = nn_preprocessing(covariates_f=covariates, factors_f=factorsf,
#                                                                         factors_test_f=factors_hatf, horizon=horizon[h])
#         y_hatff_unb = forecast_test(model=trained_model_nn, X_test_nn_f=X_test_nn, X_test_f=X_testf_unb, gamma=gammaf, ar=True, factors=_+1)
#         rmse_ar[h, _] = ivrmse(y_test=y_testf_unb, y_hat=y_hatff_unb, sc_y=sc_y)



# Forecasting using VAR(p)
for h in range(3):
    X_trainf, y_trainf, X_testf, y_testf = forecast_preprocessing(X_train_in=X_train, y_train_in=y_train,
                                                                  X_test_in=X_test, y_test_in=y_test,
                                                                  covariates_f=covariates, sc_x=sc_x, sc_y=sc_y,
                                                                  horizon=horizon[h], balanced=True)

    X_trainf_unb, y_trainf_unb, X_testf_unb, y_testf_unb = forecast_preprocessing(X_train_in=X_train_unb,
                                                                                  y_train_in=y_train_unb,
                                                                                  X_test_in=X_test_unb,
                                                                                  y_test_in=y_test_unb,
                                                                                  covariates_f=covariates, sc_x=sc_x,
                                                                                  sc_y=sc_y,
                                                                                  horizon=horizon[h], balanced=False)
    for _ in range(6):
        tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\IPCA\IPCAu_" + str(_ + 1) + "f_" + str(horizon[h]) + "h1"
        trained_model = pickle.load(open(tempdir, "rb"))

        y_hatf, factors_hatf = ipca_test(X_test_f=X_testf_unb, y_test_f=y_testf_unb, model=trained_model, n_factors=_ + 1) #Change between b and u
        gammaf, factorsf = trained_model.get_factors(label_ind=True)

        sc_f = StandardScaler()
        f_train = sc_f.fit_transform(factorsf.values.transpose())
        f_test = sc_f.transform(factors_hatf)

        factor_model = train_VAR_model(f_train=f_train)
        f_hat_nor = test_VAR_model(f_train=f_train, f_test=f_test, model=factor_model)[h]
        f_hat = sc_f.inverse_transform(f_hat_nor)

        y_hatff = forecast_test_VAR(gamma=gammaf, f_hat=f_hat, X_test_f=X_testf)
        rmse[h, _] = ivrmse(y_test=y_testf, y_hat=y_hatff, sc_y=sc_y)

        y_hatff_unb = forecast_test_VAR(gamma=gammaf, f_hat=f_hat, X_test_f=X_testf_unb)
        rmse_u[h, _] = ivrmse(y_test=y_testf_unb, y_hat=y_hatff_unb, sc_y=sc_y)
