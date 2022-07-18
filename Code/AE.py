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

X = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv')


# This is the size of our encoded representations
encoding_dim = 3  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(42,))
input_2 = keras.Input(shape=(2,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# Add inputs to decoder
merge = layers.Concatenate()([encoded, input_2])
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(42, activation='sigmoid')(merge)

# This model maps an input to its reconstruction
autoencoder = keras.Model(inputs=[input_img, input_2], outputs=decoded)

# # This model maps an input to its encoded representation
# encoder = keras.Model(input_img, encoded)
#
# # This is our encoded (32-dimensional) input
# encoded_input = keras.Input(shape=(encoding_dim,))
# # Retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # Create the decoder model
# decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# autoencoder.fit([X_train, char], X_train_adj,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(X_test, X_test_adj))