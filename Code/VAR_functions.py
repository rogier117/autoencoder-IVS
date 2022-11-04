import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Code.performance_measure import rsq, rsq_recession
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import ar_select_order, AutoReg, AutoRegResults


def train_VAR_model(f_train):
    if f_train.shape[1] > 1:
        f_train_list = pos_def_check(f_train=f_train)
        if isinstance(f_train_list, tuple):
            f_train = f_train_list[0]
            el = f_train_list[1]
            model = VAR(f_train)
            results = model.fit(maxlags=10, ic='bic')
            return results, el
        else:
            f_train = f_train_list
            model = VAR(f_train)
            results = model.fit(maxlags=10, ic='bic')
            return results
    else:
        model = ar_select_order(f_train, maxlag=10, ic='bic')
        if model.ar_lags is not None:
            p = model.ar_lags.__len__()
        else:
            p = 0
        model = AutoReg(f_train, p)
        results = model.fit()
        return results


def test_VAR_model(f_train, f_test, model, el=None):
    if f_train.shape[1] > 1:
        p = model.k_ar
    else:
        if model.ar_lags is not None:
            p = model.ar_lags.__len__()
        else:
            p = 0

    y_test = np.concatenate((f_train[-(p+20):], f_test[:-1]), axis=0)
    n = y_test.shape[0] - p + 1
    f21 = np.zeros((n,y_test.shape[1]))
    f5 = np.zeros((n, y_test.shape[1]))
    f1 = np.zeros((n, y_test.shape[1]))

    if y_test.shape[1] > 1:
        if el is not None:
            y_test = np.delete(arr=y_test, obj=el, axis=1)
        for i in range(n):
            yhat_temp = model.forecast(y_test[i:i + p], 21)
            yhat_temp = pos_def_correct(f_train=f_train, yhat_temp=yhat_temp) # CHECK TOMORROW
            f1[i] = yhat_temp[0]
            f5[i] = yhat_temp[4]
            f21[i] = yhat_temp[20]
    else:
        model.nobs = y_test.shape[0] - p
        model.model.endog = y_test
        model.model.nobs = model.nobs
        for i in range(n):
            yhat_temp = model.predict(start=i+p, end=i+p+20, dynamic=True)
            f1[i] = yhat_temp[0]
            f5[i] = yhat_temp[4]
            f21[i] = yhat_temp[20]

    f1 = f1[20:]
    f5 = f5[16:-4]
    f21 = f21[:-20]
    result_list = list((f1, f5, f21))
    return result_list

#CHECK TOMORROW
def pos_def_check(f_train):
    el = list()
    pos_def = True
    # Check which eigenvalues cause the non-positive definiteness
    eigen = np.linalg.eigvals(np.cov(f_train.transpose()))
    for i in range(f_train.shape[1]):
        if eigen[i] <= 10**-6:
            el.append(i)
            pos_def = False

    if pos_def:
        return f_train

    f_train = np.delete(arr=f_train, obj=el, axis=1)
    return f_train, el

#CHECK TOMORROW
def pos_def_correct(f_train, yhat_temp):
    if np.array_equal(f_train, pos_def_check(f_train)):
        return yhat_temp

    el = list()
    # Check which eigenvalues cause the non-positive definiteness
    eigen = np.linalg.eigvals(np.cov(f_train.transpose()))
    for i in range(f_train.shape[1]):
        if eigen[i] <= 10 ** -6:
            el.append(i)

    for _ in range(len(el)):
        value = np.mean(f_train[:,el[_]])
        insert_array = np.repeat(value, yhat_temp.shape[0])
        insert_array.shape = (yhat_temp.shape[0], 1)
        yhat_temp = np.hstack((yhat_temp[:, :el[_]], insert_array, yhat_temp[:, el[_]:]))

    return yhat_temp



if __name__ == "__main__":
    a = 0
