import math
import time
from scipy.stats import norm

import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_market_calendars as mcal  # CHANGE ALL USFEDERALHOLIDAY THINGS WITH THIS ONE!!

from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import layers


def ae_preprocessing(X, df, split=0.8):
    X = np.array(X)
    y = X.flatten()
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
    return X_train, X_test, char_train, char_test, y_train, y_test


def create_model(n_factors=3, encoder_width=21, decoder_width=21):
    # This is the size of our encoded representations
    encoding_dim = n_factors

    # This is our input image
    input_1 = keras.Input(shape=(42,))
    input_2 = keras.Input(shape=(2,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoder_width, activation='relu')(input_1)

    encoded2 = layers.Dense(encoding_dim, activation='relu')(encoded)
    # Add inputs to decoder
    merge = layers.Concatenate()([encoded2, input_2])
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(decoder_width, activation='relu')(merge)

    decoded2 = layers.Dense(1, activation='sigmoid')(decoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(inputs=[input_1, input_2], outputs=decoded2)

    # This model maps an input to its encoded representation
    # encoder = keras.Model(input_img, encoded)
    #
    # # This is our encoded (32-dimensional) input
    # encoded_input = keras.Input(shape=(encoding_dim,))
    # # Retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # # Create the decoder model
    # decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    return autoencoder


def train_ae(model, X_train, X_test, char_train, char_test, y_train, y_test, epochs=10, batch_size=42):
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


def unbalanced_test_ae(model, df_unb, X, split=0.8):
    cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
    df_unb = df_unb[df_unb.t > cut_off]

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

    y_hat = model.predict([X_test, char_test])
    return y_hat


def direct_forecast_preprocessing(X_train, X_test, char_train, char_test, y_train, y_test, horizon=1):
    gridsize = X_train.shape[1]

    X_test = np.append(X_train[-(gridsize * horizon):, :], X_test[:-(gridsize * horizon), :], axis=0)
    char_test = np.append(char_train[-(gridsize * horizon):, :], char_test[:-(gridsize * horizon), :], axis=0)

    X_train = X_train[:-(gridsize * horizon), :]
    char_train = char_train[:-(gridsize * horizon), :]

    y_train = y_train[(gridsize * horizon):]

    return X_train, X_test, char_train, char_test, y_train, y_test


def indirect_forecast_preprocessing(covariates, X_train_in, X_test_in, model, horizon=1):
    gridsize = X_train_in.shape[1]

    factor_model = keras.Model(inputs=model.inputs[0], outputs=model.layers[2].output)

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

    partial_model = keras.Model(inputs=model.layers[4].input, outputs=model.output)
    return X_train, X_test, y_train, y_test, partial_model


def indirect_forecast_train(X_train, y_train, n_epochs, batch_size):
    n_factors = y_train.shape[1]
    n_inputs = X_train.shape[1]

    input = keras.Input(shape=(n_inputs,))
    layer_1 = layers.Dense(round(n_inputs / 2), activation='relu')(input)
    layer_2 = layers.Dense(n_factors, activation='sigmoid')(layer_1)
    model = keras.Model(inputs=input, outputs=layer_2)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=True)
    return model


def indirect_forecast_test(trained_model, partial_model, X_test, char_test):
    gridsize = int(char_test.shape[0]/X_test.shape[0])

    f_test = trained_model.predict(X_test)
    f_test = np.repeat(f_test, gridsize, axis=0)
    y_hat = partial_model.predict([f_test, char_test])

    return y_hat



X = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv')
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')

split = 0.8

X_train, X_test, char_train, char_test, y_train, y_test = ae_preprocessing(X=X, df=df, split=split)
model = create_model(n_factors=3, encoder_width=21, decoder_width=21)

# Train model
trained_model = train_ae(model=model, X_train=X_train, X_test=X_test, char_train=char_train, char_test=char_test,
                         y_train=y_train, y_test=y_test, epochs=5, batch_size=42)

# Test on balanced data
y_hat = test_ae(model=trained_model, X_test=X_test, char_test=char_test)

# # Test on unbalanced data
# df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
# y_hat_unb = unbalanced_test_ae(model=trained_model, X=X, df_unb=df_unb, split=split)

# Direct forecast
# X_train, X_test, char_train, char_test, y_train, y_test = direct_forecast_preprocessing(X_train=X_train, X_test=X_test,
#                                                                                         char_train=char_train,
#                                                                                         char_test=char_test,
#                                                                                         y_train=y_train, y_test=y_test,
#                                                                                         horizon=1)
#
# model = create_model(n_factors=3, encoder_width=21, decoder_width=21)
#
# # Train model
# trained_model_f = train_ae(model=model, X_train=X_train, X_test=X_test, char_train=char_train, char_test=char_test,
#                            y_train=y_train, y_test=y_test, epochs=10, batch_size=42)
#
# # Test on balanced data
# y_hat = test_ae(model=trained_model_f, X_test=X_test, char_test=char_test)

# Indirect forecast

covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
covariates = covariates.drop(columns='Date')
X_train_f, X_test_f, y_train_f, y_test_f, partial_model = indirect_forecast_preprocessing(
    covariates=covariates,
    X_train_in=X_train,
    X_test_in=X_test,
    model=trained_model,
    horizon=1)

trained_model_if = indirect_forecast_train(X_train=X_train_f, y_train=y_train_f, n_epochs=50, batch_size=42)
y_hat_if = indirect_forecast_test(trained_model=trained_model_if, partial_model=partial_model, X_test=X_test_f,
                                  char_test=char_test)
