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
    X = X[(cut_off+1):, :]

    moneyness = np.array(df_unb.moneyness)
    tenor = np.array(df_unb.daystoex)
    char_test = np.append(moneyness[:, None], tenor[:, None], axis=1)

    X_test = np.zeros((df_unb.shape[0],X.shape[1]))

    all_t = df_unb.t.unique()
    for _ in range(len(all_t)):
        el = np.where(df_unb.t == all_t[_])[0]
        X_test[el, :] = X[_, :]

    y_hat = model.predict([X_test,char_test])
    return y_hat


def direct_forecasting_preprocessing(X_train, X_test, char_train, char_test, y_train, y_test, horizon=1):
    gridsize = X_train.shape[1]

    X_test = np.append(X_train[-(gridsize*horizon):, :], X_test[:-(gridsize*horizon), :], axis=0)
    char_test = np.append(char_train[-(gridsize * horizon):, :], char_test[:-(gridsize * horizon), :], axis=0)

    X_train = X_train[:-(gridsize*horizon), :]
    char_train = char_train[:-(gridsize*horizon), :]

    y_train = y_train[(gridsize*horizon):]

    return X_train, X_test, char_train, char_test, y_train, y_test


X = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv')
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')

split = 0.8

X_train, X_test, char_train, char_test, y_train, y_test = ae_preprocessing(X=X, df=df, split=split)
model = create_model(n_factors=3, encoder_width=21, decoder_width=21)

# Train model
trained_model = train_ae(model=model, X_train=X_train, X_test=X_test, char_train=char_train, char_test=char_test,
                         y_train=y_train, y_test=y_test, epochs=10, batch_size=42)

# Test on balanced data
y_hat = test_ae(model=trained_model, X_test=X_test, char_test=char_test)

# Test on unbalanced data
df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
y_hat_unb = unbalanced_test_ae(model=trained_model, X=X, df_unb=df_unb, split=split)


# Direct forecast
X_train, X_test, char_train, char_test, y_train, y_test = direct_forecasting_preprocessing(X_train=X_train,
                                                                                           X_test=X_test,
                                                                                           char_train=char_train,
                                                                                           char_test=char_test,
                                                                                           y_train=y_train,
                                                                                           y_test=y_test,
                                                                                           horizon=1)

model = create_model(n_factors=3, encoder_width=21, decoder_width=21)

# Train model
trained_model_f = train_ae(model=model, X_train=X_train, X_test=X_test, char_train=char_train, char_test=char_test,
                           y_train=y_train, y_test=y_test, epochs=10, batch_size=42)

# Test on balanced data
y_hat = test_ae(model=trained_model_f, X_test=X_test, char_test=char_test)
