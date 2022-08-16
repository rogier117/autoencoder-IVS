import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Code.performance_measure import rsq
import pickle


def pca_preprocessing(X):
    # distributing the dataset into two components X and Y
    X = X.values
    X_train, X_test = train_test_split(X, test_size=1006, shuffle=False)

    # performing preprocessing part
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return sc, X_train, X_test


def train_model(n_factors, X_train):
    pca = PCA(n_components=n_factors)
    # train PCA on normalized X_train
    pca.fit(X_train)
    f_train = pca.transform(X_train)
    return pca, f_train


def test_model(model, X_test):
    # compute factors
    f_test = model.transform(X_test)
    # return original normalized data
    X_hat_nor = model.inverse_transform(f_test)
    return f_test, X_hat_nor


def unbalanced_test_model(df_unb, X_hat, split=0.8):
    # Takes about 2 minutes
    cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
    df_unb = df_unb[df_unb.t > cut_off].reset_index(drop=True)
    # start t at 0 for convenience
    df_unb['t'] = df_unb.t - (cut_off + 1)

    # set tau and k on the same scale for kernel smoothing
    df_unb['tau_nor'] = (df_unb.daystoex - 10) / 242
    df_unb['k_nor'] = (df_unb.moneyness - 0.9) / 0.4

    tenors = np.array([10, 21, 63, 126, 189, 252])
    tenors = (tenors - 10) / 242
    moneyness = np.array([0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3])
    moneyness = (moneyness - 0.9) / 0.4
    grid = np.meshgrid(tenors, moneyness)
    gridsize = len(tenors) * len(moneyness)
    tenors = grid[0].flatten()
    moneyness = grid[1].flatten()

    # perform kernel smoothing
    b = 5 / 100
    y_hat = np.zeros(df_unb.shape[0])
    for _ in tqdm(range(df_unb.shape[0]), desc='option'):
        tau_temp = np.repeat(df_unb.tau_nor[_], gridsize)
        k_temp = np.repeat(df_unb.k_nor[_], gridsize)
        euc_sq = (tau_temp - tenors) ** 2 + (k_temp - moneyness) ** 2
        # kernel is the weight that each point gets
        kernel = np.exp(- euc_sq / (b ** 2))
        y_hat[_] = np.dot(X_hat[df_unb.t[_], :], kernel) / np.sum(kernel)
    return y_hat


def forecast_pre_processing(covariates, f_train, f_test, horizon):
    # Since we use the lag of factors, we can not use data before the start of the sample, and the final h observations
    X_train = covariates.values
    X_train = X_train[21:, :]

    y_train = f_train[horizon:, :]
    X_test = X_train[len(y_train):-horizon, :]
    X_test = np.append(X_test, np.append(f_train[-horizon:, :], f_test[:-horizon, :], axis=0), axis=1)
    X_train = X_train[:len(y_train), :]
    X_train = np.append(X_train, f_train[:-horizon, :], axis=1)

    # normalize all covariates
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train)
    X_test = sc_x.transform(X_test)

    y_test = f_test # Combination of end of f_train and f_test

    # normalize the factor values
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    return X_train, X_test, y_train, y_test, sc_y, sc_x


def forecast_train(X_train, y_train, n_epochs=10, batch_size=64):
    # make and train neural network to forecast factors
    n_factors = y_train.shape[1]
    n_inputs = X_train.shape[1]

    input = keras.Input(shape=(n_inputs,))
    layer_1 = layers.Dense(round(n_inputs/2), activation='relu')(input)
    layer_2 = layers.Dense(n_factors, activation='tanh')(layer_1)
    model = keras.Model(inputs=input, outputs=layer_2)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=True)
    return model


def forecast_test(pca, model, X_test, scfy, sc):
    # covariates -> normalized factors
    y_hat = model.predict(X_test)
    # normalized factors -> factors
    y_hat = scfy.inverse_transform(y_hat)
    # factors -> normalized X
    X_hat = pca.inverse_transform(y_hat)
    return X_hat


# # Pre-processing
# sc, X_train, X_test = pca_preprocessing(X=X)
# pca, f_train = train_model(n_factors=3, X_train=X_train)
# f_test, X_hat = test_model(model=pca, X_test=X_test)
#
# # Test performance on unbalanced data
# df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
# y_hat = unbalanced_test_model(df_unb=df_unb, X_hat=X_hat, split=0.8)
#
#
# # Forecasting
# covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
# covariates = covariates.drop(columns='Date')
# X_train_f, X_test_f, y_train_f, y_test_f, scfy, scfx = forecast_pre_processing(covariates=covariates, f_train=f_train,
#                                                                                f_test=f_test, horizon=1)
# fmodel = forecast_train(X_train=X_train_f, y_train=y_train_f, n_epochs=100, batch_size=64)
# X_hat_f = forecast_test(pca=pca, model=fmodel, X_test=X_test_f, scfy=scfy, sc=sc)

# importing or loading the dataset
X = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv')
df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
split = 0.8
cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
df_unb_test = df_unb[df_unb.t > cut_off].reset_index(drop=True)

r2 = np.zeros(6)
r2_u = np.zeros(6)
sc, X_train, X_test = pca_preprocessing(X=X)
for _ in range(6):
    pca, f_train = train_model(n_factors=_+1, X_train=X_train)
    f_test, X_hat = test_model(model=pca, X_test=X_test)
    r2[_] = rsq(y_test=X_test, y_hat=X_hat, sc_y=sc)

    X_hat = sc.inverse_transform(X_hat)
    y_hat = unbalanced_test_model(df_unb=df_unb, X_hat=X_hat, split=0.8)
    r2_u[_] = rsq(y_test=df_unb_test.IV, y_hat=y_hat, sc_y=None)

    # tempdir = r"D:\Master Thesis\autoencoder-IVS\Models\Modelling\PCA\PCA_" + str(_+1) + "f_0h.sav"
    # pickle.dump(pca, open(tempdir, 'wb'))
