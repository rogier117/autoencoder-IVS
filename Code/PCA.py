import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    pca.fit(X_train)
    f_train = pca.transform(X_train)
    return pca, f_train


def test_model(sc, model, X_test):
    f_test = model.transform(X_test)
    X_hat_nor = model.inverse_transform(f_test)
    X_hat = sc.inverse_transform(X_hat_nor)
    return f_test, X_hat


def unbalanced_test_model(df_unb, X_hat, split=0.8):
    # Takes about 2 minutes
    cut_off = df_unb.t.unique()[round(len(df_unb.t.unique()) * split)]
    df_unb = df_unb[df_unb.t > cut_off].reset_index(drop=True)
    df_unb['t'] = df_unb.t - (cut_off + 1)

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

    y_test = f_test # Combination of end of f_train and f_test

    sc = StandardScaler()
    y_train = sc.fit_transform(y_train)
    y_test = sc.transform(y_test)
    return X_train, X_test, y_train, y_test, sc


def forecast_train(X_train, y_train, n_epochs=10, batch_size=64):
    n_factors = y_train.shape[1]
    n_inputs = X_train.shape[1]

    input = keras.Input(shape=(n_inputs,))
    layer_1 = layers.Dense(round(n_inputs/2), activation='relu')(input)
    layer_2 = layers.Dense(n_factors, activation='sigmoid')(layer_1)
    model = keras.Model(inputs=input, outputs=layer_2)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train,
              epochs=n_epochs,
              batch_size=batch_size,
              shuffle=True)
    return model


def forecast_test(pca, model, X_test, scf, sc):
    y_hat = model.predict(X_test)
    y_hat = scf.inverse_transform(y_hat)
    X_hat = pca.inverse_transform(y_hat)
    X_hat = sc.inverse_transform(X_hat)
    return X_hat


# importing or loading the dataset
X = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv')

# Pre-processing
sc, X_train, X_test = pca_preprocessing(X=X)
pca, f_train = train_model(n_factors=3, X_train=X_train)
f_test, X_hat = test_model(sc=sc, model=pca, X_test=X_test)

# Calculate total R^2 as described in paper
numerator = np.sum((sc.inverse_transform(X_test) - X_hat) ** 2)
denominator = np.sum((sc.inverse_transform(X_test) - np.mean(sc.inverse_transform(X_test))) ** 2)
total_Rsq = 1 - numerator/denominator

# Test performance on unbalanced data
# df_unb = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
# y_hat = unbalanced_test_model(df_unb=df_unb, X_hat=X_hat, split=0.8)


# Forecasting
covariates = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\covariates.csv')
covariates = covariates.drop(columns='Date')
X_train, X_test, y_train, y_test, scf = forecast_pre_processing(covariates=covariates, f_train=f_train, f_test=f_test,
                                                                horizon=1)
fmodel = forecast_train(X_train=X_train, y_train=y_train, n_epochs=100, batch_size=64)
X_hat_f = forecast_test(pca=pca, model=fmodel, X_test=X_test, scf=scf, sc=sc)



# # Calculate daily total R^2
# Rsq_array = np.zeros(X_hat_nor.shape[0])
# for i in range(X_hat_nor.shape[0]):
#     temp_hat = X_hat_nor[i,:]
#     temp_test = X_test_nor[i,:]
#     temp_num = np.sum((temp_test - temp_hat) ** 2)
#     temp_denom = np.sum(temp_test ** 2)
#     Rsq_array[i] = 1 - temp_num / temp_denom
#
# # Plotting with dates: import dataset with dates
# SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
# SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
# plt.plot(SPX.Date[-len(Rsq_array):],Rsq_array)
