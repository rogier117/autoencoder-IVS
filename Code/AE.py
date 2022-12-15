import math
import time
from scipy.stats import norm

import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_market_calendars as mcal  # CHANGE ALL USFEDERALHOLIDAY THINGS WITH THIS ONE!!
from sklearn.preprocessing import StandardScaler

from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import layers
from Code.performance_measure import rsq, rsq_recession, ivrmse
from Code.VAR_functions import train_VAR_model, test_VAR_model

from statsmodels.tsa.stattools import adfuller


def ae_preprocessing(X, df, split):
    X = np.array(X)

    # y consists of all IV's available
    y = X.flatten()

    # characters are present in the df
    moneyness = np.array(df.moneyness)
    tenor = np.array(df.daystoex)
    char = np.append(moneyness[:, None], tenor[:, None], axis=1)

    X = np.repeat(X, X.shape[1], axis=0)

    cut_off = df.t.unique()[round(len(df.t.unique()) * split)]
    X_train = X[df.t <= cut_off, :]
    X_test = X[df.t > cut_off, :]
    char_train = char[df.t <= cut_off, :]
    char_test = char[df.t > cut_off, :]
    y_train = y[df.t <= cut_off]
    y_test = y[df.t > cut_off]

    sc_x = StandardScaler()
    X_train= sc_x.fit_transform(X_train)
    X_test= sc_x.transform(X_test)

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1,1)).reshape(-1,)
    y_test = sc_y.transform(y_test.reshape(-1,1)).reshape(-1,)

    sc_c = StandardScaler()
    char_train = sc_c.fit_transform(char_train)
    char_test = sc_c.transform(char_test)

    return X_train, X_test, char_train, char_test, y_train, y_test, sc_x, sc_y, sc_c


def create_model(n_factors, encoder_width, decoder_width):
    # Set seed
    tf.random.set_seed(1234)
    # This is the size of our encoded representations
    encoding_dim = n_factors

    # This is our input image
    input_1 = keras.Input(shape=(42,))
    input_2 = keras.Input(shape=(2,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoder_width, activation='relu')(input_1)

    encoded_temp = layers.Dense(encoder_width, activation='relu')(encoded)

    encoded2 = layers.Dense(encoding_dim, activation='relu')(encoded_temp)
    # Add inputs to decoder
    merge = layers.Concatenate()([encoded2, input_2])
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(decoder_width, activation='relu')(merge)

    decoded_temp = layers.Dense(decoder_width, activation='relu')(decoded)

    decoded2 = layers.Dense(1, activation='linear')(decoded_temp)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(inputs=[input_1, input_2], outputs=decoded2)

    return autoencoder


def train_ae(model, X_train, X_test, char_train, char_test, y_train, y_test, epochs, batch_size):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit([X_train, char_train], y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=([X_test, char_test], y_test))
    return model


def test_ae(model, X_test, char_test):
    y_hat = model.predict([X_test, char_test])
    return y_hat


def unbalanced_test_ae(model, df_unb, X, sc_x, sc_c, split, rec=False):
    cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
    df_unb = df_unb[df_unb.t > cut_off]

    # Remove from recession onwards
    if rec:
        alldates = df_unb.t - 4030
        df_unb = df_unb[alldates < 522]

    X = np.array(X)
    X = X[(cut_off + 1):, :]

    moneyness = np.array(df_unb.moneyness)
    tenor = np.array(df_unb.daystoex)
    char_test = np.append(moneyness[:, None], tenor[:, None], axis=1)

    X_test = np.zeros((df_unb.shape[0], X.shape[1]))

    all_t = df_unb.t.unique()
    for _ in range(len(all_t)):
        el = np.where(df_unb.t == all_t[_])[0]
        X_test[el, :] = X[_, :]

    X_test = sc_x.transform(X_test)
    char_test = sc_c.transform(char_test)

    y_hat = model.predict([X_test, char_test])
    return y_hat


def direct_forecast_preprocessing(X_train, X_test, char_train, char_test, y_train, y_test, horizon):
    gridsize = X_train.shape[1]

    X_test = np.append(X_train[-(gridsize * horizon):, :], X_test[:-(gridsize * horizon), :], axis=0)
    char_test = np.append(char_train[-(gridsize * horizon):, :], char_test[:-(gridsize * horizon), :], axis=0)

    X_train = X_train[:-(gridsize * horizon), :]
    char_train = char_train[:-(gridsize * horizon), :]

    y_train = y_train[(gridsize * horizon):]

    return X_train, X_test, char_train, char_test, y_train, y_test[:]


def indirect_forecast_preprocessing(covariates, X_train_in, X_test_in, model, horizon):
    gridsize = X_train_in.shape[1]

    factor_model = keras.Model(inputs=model.inputs[0], outputs=model.layers[3].output)

    X_train_in = X_train_in[0::gridsize, :]
    X_test_in = X_test_in[0::gridsize, :]

    f_train = factor_model(X_train_in)
    f_train = np.array(f_train)

    f_test = factor_model(X_test_in)
    f_test = np.array(f_test)

    y_train = f_train[horizon:, :]
    y_test = f_test

    X_train = covariates.values
    X_train = X_train[21:, :]

    X_test = X_train[len(y_train):-horizon, :]
    X_test = np.append(X_test, np.append(f_train[-horizon:, :], f_test[:-horizon, :], axis=0), axis=1)
    X_train = X_train[:len(y_train), :]
    X_train = np.append(X_train, f_train[:-horizon, :], axis=1)

    sc_fx = StandardScaler()
    X_train = sc_fx.fit_transform(X_train)
    X_test = sc_fx.transform(X_test)

    sc_fy = StandardScaler()
    y_train = sc_fy.fit_transform(y_train)
    y_test = sc_fy.transform(y_test)

    partial_model = keras.Model(inputs=model.layers[6].input, outputs=model.output, name='partial')
    return X_train, X_test, y_train, y_test, partial_model, sc_fx, sc_fy


def indirect_forecast_train(X_train, y_train, n_epochs, batch_size, width):
    # Set seed
    tf.random.set_seed(1234)

    n_factors = y_train.shape[1]
    n_inputs = X_train.shape[1]

    input = keras.Input(shape=(n_inputs,))
    layer_1 = layers.Dense(width, activation='relu')(input)
    layer_15 = layers.Dense(width, activation='relu')(layer_1)
    layer_2 = layers.Dense(n_factors, activation='linear')(layer_15)
    model = keras.Model(inputs=input, outputs=layer_2)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=True)
    return model


def indirect_forecast_test(trained_model, partial_model, X_test, char_test, sc_fy):
    gridsize = int(char_test.shape[0] / X_test.shape[0])

    f_test = trained_model.predict(X_test)
    f_test = sc_fy.inverse_transform(f_test)
    f_test = np.repeat(f_test, gridsize, axis=0)
    # y_hat = partial_model.predict([f_test, char_test])
    y_hat = partial_model.predict(np.concatenate((f_test, char_test), axis=1))

    return y_hat


def indirect_forecast_test_unb(trained_model, partial_model, df_unb, split, X_test, sc_fy, sc_c, rec=False, ar=False, sc_fx=None):
    f_test = trained_model.predict(X_test)
    f_test = sc_fy.inverse_transform(f_test)

    if ar:
        X_test_temp = sc_fx.inverse_transform(X_test)
        f_test = X_test_temp[:, -f_test.shape[1]:]

    cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
    df_unb = df_unb[df_unb.t > cut_off]

    if rec:
        alldates = df_unb.t - 4030
        df_unb = df_unb[alldates < 522]

    moneyness = np.array(df_unb.moneyness)
    tenor = np.array(df_unb.daystoex)
    char_test = np.append(moneyness[:, None], tenor[:, None], axis=1)

    f_test_final = np.zeros((df_unb.shape[0], f_test.shape[1]))

    all_t = df_unb.t.unique()
    for _ in range(len(all_t)):
        el = np.where(df_unb.t == all_t[_])[0]
        f_test_final[el, :] = f_test[_, :]

    char_test = sc_c.transform(char_test)

    # y_hat = partial_model.predict([f_test_final, char_test])
    y_hat = partial_model.predict(np.concatenate((f_test_final, char_test), axis=1))
    return y_hat


def indirect_forecast_test_VAR(trained_model, X_train, X_test, df_unb, char_test, nlayers, sc_c, balanced):
    factor_model = keras.Model(inputs=trained_model.inputs[0], outputs=trained_model.layers[nlayers+1].output)
    partial_model = keras.Model(inputs=trained_model.layers[nlayers+4].input, outputs=trained_model.output, name='partial')

    gridsize = X_train.shape[1]

    X_train = X_train[0::gridsize, :]
    X_test = X_test[0::gridsize, :]

    f_train = factor_model(X_train)
    f_train = np.array(f_train)

    f_test = factor_model(X_test)
    f_test = np.array(f_test)

    sc_f = StandardScaler()
    f_train_nor = sc_f.fit_transform(f_train)
    f_test_nor = sc_f.transform(f_test)

    model = train_VAR_model(f_train=f_train_nor)
    el = None
    if isinstance(model, tuple):
        el = model[1]
        model = model[0]
    result_list = test_VAR_model(f_train=f_train_nor, f_test=f_test_nor, model=model, el=el)
    yhat_list = list()

    if balanced:
        for h in range(3):
            f_hat_temp = result_list[h]
            f_hat = sc_f.inverse_transform(f_hat_temp)
            f_hat = np.repeat(f_hat, gridsize, axis=0)
            y_hat = partial_model.predict(np.concatenate((f_hat, char_test), axis=1))
            yhat_list.append(y_hat)

    else:
        split = 0.8
        cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
        df_unb = df_unb[df_unb.t > cut_off]
        for h in range(3):
            f_hat_temp = result_list[h]
            f_hat = sc_f.inverse_transform(f_hat_temp)
            moneyness = np.array(df_unb.moneyness)
            tenor = np.array(df_unb.daystoex)
            char_test = np.append(moneyness[:, None], tenor[:, None], axis=1)
            char_test = sc_c.transform(char_test)

            f_hat_final = np.zeros((df_unb.shape[0], f_hat.shape[1]))

            all_t = df_unb.t.unique()
            for _ in range(len(all_t)):
                el = np.where(df_unb.t == all_t[_])[0]
                f_hat_final[el, :] = f_hat[_, :]
            y_hat = partial_model.predict(np.concatenate((f_hat_final, char_test), axis=1))
            yhat_list.append(y_hat)
    return yhat_list


def bootstrap_data(bs, X_train, char_train, y_train):
    gridsize = X_train.shape[1]
    options = X_train.shape[0]
    days = options/gridsize

    X_train_out = list()
    X_test_out = list()

    char_train_out = list()
    char_test_out = list()

    y_train_out = list()
    y_test_out = list()

    for _ in range(bs):
        cut_begin = round((days * _) / bs) * gridsize
        cut_end = round((days * (_ + 1)) / bs) * gridsize

        X_test_out.append(X_train[cut_begin:cut_end, :])
        y_test_out.append(y_train[cut_begin:cut_end])
        char_test_out.append(char_train[cut_begin:cut_end, :])
        # Selection of elements for train
        X_train_begin = X_train[:cut_begin, :]
        X_train_end = X_train[cut_end:, :]
        X_train_out.append(np.append(X_train_begin, X_train_end, axis=0))

        char_train_begin = char_train[:cut_begin, :]
        char_train_end = char_train[cut_end:, :]
        char_train_out.append(np.append(char_train_begin, char_train_end, axis=0))

        y_train_begin = y_train[:cut_begin]
        y_train_end = y_train[cut_end:]
        y_train_out.append(np.append(y_train_begin, y_train_end))

    return X_train_out, X_test_out, char_train_out, char_test_out, y_train_out, y_test_out


def bootstrap(X_train, char_train, y_train, bs, sc_y, epochs, batch_size, width):
    X_train_f = X_train[:, :]
    char_train_f = char_train[:, :]
    y_train_f = y_train[:]

    X_train_temp, X_test_temp, char_train_temp, char_test_temp, y_train_temp, y_test_temp = bootstrap_data(bs=bs, X_train=X_train_f, char_train=char_train_f, y_train=y_train_f)
    rmse_temp = np.zeros(bs)

    for i in range(bs):
        model = create_model(n_factors=3, encoder_width=width, decoder_width=width)
        trained_model = train_ae(model=model, X_train=X_train_temp[i], X_test=X_test_temp[i], char_train=char_train_temp[i],
                                 char_test=char_test_temp[i], y_train=y_train_temp[i], y_test=y_test_temp[i],
                                 epochs=epochs, batch_size=batch_size)
        y_hat = test_ae(model=trained_model, X_test=X_test_temp[i], char_test=char_test_temp[i])
        rmse_temp[i] = ivrmse(y_test=y_test_temp[i], y_hat=y_hat, sc_y=sc_y)
    rmse = np.mean(rmse_temp)
    return rmse


X = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv')
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')

split = 0.8

X_train, X_test, char_train, char_test, y_train, y_test, sc_x, sc_y, sc_c = ae_preprocessing(X=X, df=df, split=split)


# # Bootstrapping
# epochs = np.array([100, 250])
# batch = np.array([42, 84])
# width = np.array([32, 64, 128])
# grid = np.array(np.meshgrid(epochs, batch, width)).T.reshape(-1,3)
# rmse = np.zeros(grid.shape[0])
#
# for _ in range(grid.shape[0]):
#     rmse[_] = bootstrap(X_train=X_train, char_train=char_train, y_train=y_train, bs=5, sc_y=sc_y, epochs=grid[_, 0], batch_size=grid[_, 1], width=grid[_, 2])
# # Optimal parameters: activation=ReLU, epochs=100, batch=84, width=64

# # Train and test model
df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
df_unb_test = df_unb[df_unb.t > cut_off]

# rmse = np.zeros(6)
# rmse_u = np.zeros(6)
# for _ in range(6):
#     # model = create_model(n_factors=_+1, encoder_width=64, decoder_width=64)
#     #
#     # trained_model = train_ae(model=model, X_train=X_train, X_test=X_test, char_train=char_train, char_test=char_test,
#     #                          y_train=y_train, y_test=y_test, epochs=100, batch_size=84)
#     tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE2_" + str(_ + 1) + "f_0h"
#     trained_model = keras.models.load_model(tempdir)
#     y_hat = test_ae(model=trained_model, X_test=X_test, char_test=char_test)
#     rmse[_] = ivrmse(y_test=y_test, y_hat=y_hat, sc_y=sc_y)
#
#     # Unbalanced testing
#     y_hat_unb = unbalanced_test_ae(model=trained_model, df_unb=df_unb, sc_x=sc_x, sc_c=sc_c, X=X, split=split)
#     rmse_u[_] = ivrmse(y_test=sc_y.transform(df_unb_test.IV.values.reshape(-1, 1)), y_hat=y_hat_unb, sc_y=sc_y)
#
#     # tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE1_" + str(_ + 1) + "f_0h"
#     # trained_model.save(tempdir)

# # Factor interpretation AE1 3-factor model
# tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE1_3f_0h"
# trained_model = keras.models.load_model(tempdir)
#
# factor_model = keras.Model(inputs=trained_model.inputs[0], outputs=trained_model.layers[2].output)
# gridsize = X_train.shape[1]
# X_train_factor = X_train[0::gridsize, :]
# X_test_factor = X_test[0::gridsize, :]
# X_factor = np.concatenate((X_train_factor, X_test_factor), axis=0)
# factors = np.array(factor_model(X_factor))


# # Factor interpretation AE2 3-factor model
# tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE2_3f_0h"
# trained_model = keras.models.load_model(tempdir)
#
# factor_model = keras.Model(inputs=trained_model.inputs[0], outputs=trained_model.layers[3].output)
# gridsize = X_train.shape[1]
# X_train_factor = X_train[0::gridsize, :]
# X_test_factor = X_test[0::gridsize, :]
# X_factor = np.concatenate((X_train_factor, X_test_factor), axis=0)
# factors = np.array(factor_model(X_factor))


# # Direct forecast
#
# rmse = np.zeros((3,6))
# rmse_u = np.zeros((3,6))
#
# rmse_rec = np.zeros((3, 6))
# rmse_ar = np.zeros((3, 6))
#
# horizon = np.array([1, 5, 21])
#
# for h in range(3):
#     X_train_f, X_test_f, char_train_f, char_test_f, y_train_f, y_test_f = direct_forecast_preprocessing(X_train=X_train,
#                                                                                                         X_test=X_test,
#                                                                                                         char_train=char_train,
#                                                                                                         char_test=char_test,
#                                                                                                         y_train=y_train,
#                                                                                                         y_test=y_test,
#                                                                                                         horizon=horizon[h])
#     for _ in range(6):
#         model = create_model(n_factors=_+1, encoder_width=64, decoder_width=64)
#         trained_model_f = train_ae(model=model, X_train=X_train_f, X_test=X_test_f, char_train=char_train_f,
#                                    char_test=char_test_f, y_train=y_train_f, y_test=y_test_f, epochs=100, batch_size=84)
#
#         y_hat_f = test_ae(model=trained_model_f, X_test=X_test_f, char_test=char_test_f)
#         rmse[h, _] = ivrmse(y_test=y_test_f, y_hat=y_hat_f, sc_y=sc_y)
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\AE\AE2d_" + str(_ + 1) + "f_" + str(horizon[h]) +"h"
#         trained_model_f.save(tempdir)

# # Direct forecast test
# for h in range(3):
#     X_train_f, X_test_f, char_train_f, char_test_f, y_train_f, y_test_f = direct_forecast_preprocessing(X_train=X_train,
#                                                                                                         X_test=X_test,
#                                                                                                         char_train=char_train,
#                                                                                                         char_test=char_test,
#                                                                                                         y_train=y_train,
#                                                                                                         y_test=y_test,
#                                                                                                         horizon=horizon[h])
#     X_f = X.values
#     z = np.zeros((horizon[h], X_f.shape[1]))
#     X_f = np.concatenate((z, X_f), axis=0)
#     X_f = X_f[:-horizon[h], :]
#     for _ in range(6):
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\AE\AE2d_" + str(_ + 1) + "f_" + str(horizon[h]) +"h"
#         trained_model_f = keras.models.load_model(tempdir)
#
#         y_hat_f = test_ae(model=trained_model_f, X_test=X_test_f, char_test=char_test_f)
#         rmse[h, _] = ivrmse(y_test=y_test_f, y_hat=y_hat_f, sc_y=sc_y)
#
#         y_hat_f_u = unbalanced_test_ae(model=trained_model_f, df_unb=df_unb, X=X_f, sc_x=sc_x, sc_c=sc_c, split=0.8)
#         rmse_u[h, _] = ivrmse(y_test=sc_y.transform(df_unb_test.IV.values.reshape(-1, 1)), y_hat=y_hat_f_u, sc_y=sc_y)

# # Direct forecast test until recession
# alldates = df_unb_test.t - 4030
# df_unb_test1 = df_unb_test[alldates < 522]
# for h in range(3):
#     X_train_f, X_test_f, char_train_f, char_test_f, y_train_f, y_test_f = direct_forecast_preprocessing(X_train=X_train,
#                                                                                                         X_test=X_test,
#                                                                                                         char_train=char_train,
#                                                                                                         char_test=char_test,
#                                                                                                         y_train=y_train,
#                                                                                                         y_test=y_test,
#                                                                                                         horizon=horizon[h])
#     X_f = X.values
#     z = np.zeros((horizon[h], X_f.shape[1]))
#     X_f = np.concatenate((z, X_f), axis=0)
#     X_f = X_f[:-horizon[h], :]
#     for _ in range(6):
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\AE\AE2d_" + str(_ + 1) + "f_" + str(horizon[h]) +"h"
#         trained_model_f = keras.models.load_model(tempdir)
#         y_hat_f_u = unbalanced_test_ae(model=trained_model_f, df_unb=df_unb, X=X_f, sc_x=sc_x, sc_c=sc_c, split=0.8, rec=True)
#         rmse_rec[h, _] = ivrmse(y_test=sc_y.transform(df_unb_test1.IV.values.reshape(-1, 1)), y_hat=y_hat_f_u, sc_y=sc_y)


rmse = np.zeros((3,6))
rmse_u = np.zeros((3,6))

rmse_rec = np.zeros((3, 6))
rmse_ar = np.zeros((3, 6))

horizon = np.array([1, 5, 21])

covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
covariates = covariates.drop(columns='Date')

# Indirect forecast
# for _ in range(6):
#     tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE2_" + str(_ + 1) + "f_0h"
#     trained_model = keras.models.load_model(tempdir)
#     for h in range(3):
#         X_train_if, X_test_if, y_train_if, y_test_if, partial_model, sc_fx, sc_fy = indirect_forecast_preprocessing(
#             covariates=covariates,
#             X_train_in=X_train,
#             X_test_in=X_test,
#             model=trained_model,
#             horizon=horizon[h])
#
#         trained_model_if = indirect_forecast_train(X_train=X_train_if, y_train=y_train_if, n_epochs=100, batch_size=84, width=64)
#         y_hat_if = indirect_forecast_test(trained_model=trained_model_if, partial_model=partial_model, X_test=X_test_if,
#                                           char_test=char_test)
#
#         rmse[h, _] = ivrmse(y_test=y_test, y_hat=y_hat_if, sc_y=sc_y)
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\AE\AE2i_" + str(_ + 1) + "f_" + str(
#             horizon[h]) + "h"
#         trained_model_if.save(tempdir)


# Indirect forecast test
for _ in range(6):
    tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE2_" + str(_ + 1) + "f_0h"
    trained_model = keras.models.load_model(tempdir)
    trained_model._name = 'full'
    for h in range(3):
        X_train_if, X_test_if, y_train_if, y_test_if, partial_model, sc_fx, sc_fy = indirect_forecast_preprocessing(
            covariates=covariates,
            X_train_in=X_train,
            X_test_in=X_test,
            model=trained_model,
            horizon=horizon[h])

        tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\AE\AE2i_" + str(_ + 1) + "f_" + str(
            horizon[h]) + "h"
        trained_model_if = keras.models.load_model(tempdir)
        trained_model_if._name = 'indirect'

        y_hat_if = indirect_forecast_test(trained_model=trained_model_if, partial_model=partial_model, X_test=X_test_if,
                                          char_test=char_test, sc_fy=sc_fy)

        rmse[h, _] = ivrmse(y_test=y_test, y_hat=y_hat_if, sc_y=sc_y)

        y_hat_if_u = indirect_forecast_test_unb(trained_model=trained_model_if, partial_model=partial_model,
                                                df_unb=df_unb, split=0.8, X_test=X_test_if, sc_fy=sc_fy, sc_c=sc_c)

        rmse_u[h, _] = ivrmse(y_test=sc_y.transform(df_unb_test.IV.values.reshape(-1, 1)), y_hat=y_hat_if_u, sc_y=sc_y)


# # Indirect forecast test until recession
# alldates = df_unb_test.t - 4030
# df_unb_test1 = df_unb_test[alldates < 522]
# for _ in range(6):
#     tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE2_" + str(_ + 1) + "f_0h"
#     trained_model = keras.models.load_model(tempdir)
#     trained_model._name = 'full'
#     for h in range(3):
#         X_train_if, X_test_if, y_train_if, y_test_if, partial_model, sc_fx, sc_fy = indirect_forecast_preprocessing(
#             covariates=covariates,
#             X_train_in=X_train,
#             X_test_in=X_test,
#             model=trained_model,
#             horizon=horizon[h])
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\AE\AE2i_" + str(_ + 1) + "f_" + str(
#             horizon[h]) + "h"
#         trained_model_if = keras.models.load_model(tempdir)
#         trained_model_if._name = 'indirect'
#
#         y_hat_if_u = indirect_forecast_test_unb(trained_model=trained_model_if, partial_model=partial_model,
#                                                 df_unb=df_unb, split=0.8, X_test=X_test_if, sc_fy=sc_fy, sc_c=sc_c, rec=True)
#
#         rmse_rec[h, _] = ivrmse(y_test=sc_y.transform(df_unb_test1.IV.values.reshape(-1, 1)), y_hat=y_hat_if_u, sc_y=sc_y)


# # Indirect forecast test ar term
# for _ in range(6):
#     tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE2_" + str(_ + 1) + "f_0h"
#     trained_model = keras.models.load_model(tempdir)
#     trained_model._name = 'full'
#     for h in range(3):
#         X_train_if, X_test_if, y_train_if, y_test_if, partial_model, sc_fx, sc_fy = indirect_forecast_preprocessing(
#             covariates=covariates,
#             X_train_in=X_train,
#             X_test_in=X_test,
#             model=trained_model,
#             horizon=horizon[h])
#
#         tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Forecasting\AE\AE2i_" + str(_ + 1) + "f_" + str(
#             horizon[h]) + "h"
#         trained_model_if = keras.models.load_model(tempdir)
#         trained_model_if._name = 'indirect'
#
#         y_hat_if_u = indirect_forecast_test_unb(trained_model=trained_model_if, partial_model=partial_model,
#                                                 df_unb=df_unb, split=0.8, X_test=X_test_if, sc_fy=sc_fy, sc_c=sc_c, ar=True, sc_fx=sc_fx)
#
#         rmse_ar[h, _] = ivrmse(y_test=sc_y.transform(df_unb_test.IV.values.reshape(-1, 1)), y_hat=y_hat_if_u, sc_y=sc_y)


# # Forecast using VAR(p)
# for _ in range(6):
#     tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\AE\AE2_" + str(_ + 1) + "f_0h"
#     trained_model = keras.models.load_model(tempdir)
#     trained_model._name = 'full'
#
#     yhat_list = indirect_forecast_test_VAR(trained_model=trained_model, X_train=X_train, X_test=X_test, df_unb=df_unb,
#                                            char_test=char_test, nlayers=2, sc_c=sc_c, balanced=True)
#     for h in range(3):
#         rmse[h, _] = ivrmse(y_test=y_test, y_hat=yhat_list[h], sc_y=sc_y)
#
#     yhat_u_list = indirect_forecast_test_VAR(trained_model=trained_model, X_train=X_train, X_test=X_test, df_unb=df_unb,
#                                              char_test=char_test, nlayers=2, sc_c=sc_c, balanced=False)
#     for h in range(3):
#         rmse_u[h, _] = ivrmse(y_test=sc_y.transform(df_unb_test.IV.values.reshape(-1, 1)), y_hat=yhat_u_list[h], sc_y=sc_y)
