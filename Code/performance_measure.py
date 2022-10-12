import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rsq(y_test, y_hat, sc_y):

    if y_test.shape.__len__() == 1 and isinstance(y_test, pd.Series):
        y_test = y_test.values.reshape(-1, 1)
    elif y_test.shape.__len__() == 1:
        y_test = y_test.reshape(-1, 1)
    if y_hat.shape.__len__() == 1:
        y_hat = y_hat.reshape(-1, 1)

    if sc_y is not None:
        y_test = sc_y.inverse_transform(y_test)
        y_hat = sc_y.inverse_transform(y_hat)

    y_test = y_test.flatten()
    y_hat = y_hat.flatten()

    num = np.sum((y_test - y_hat)**2)
    denom = np.sum(y_test**2)
    rsquared = 1 - num/denom
    return rsquared


def rsq_recession(y_test, y_hat, sc_y, horizon):
    if y_test.shape.__len__() == 1 and isinstance(y_test, pd.Series):
        y_test = y_test.values.reshape(-1, 1)
    elif y_test.shape.__len__() == 1:
        y_test = y_test.reshape(-1, 1)
    if y_hat.shape.__len__() == 1:
        y_hat = y_hat.reshape(-1, 1)

    if sc_y is not None:
        y_test = sc_y.inverse_transform(y_test)
        y_hat = sc_y.inverse_transform(y_hat)

    y_test = y_test.flatten()
    y_hat = y_hat.flatten()

    # COVID recession from 2020-02-01 to 2020-04-01. In test set: 522 to 563
    y_test = np.concatenate((y_test[:42*522], y_test[42*(564+horizon):]))
    y_hat = np.concatenate((y_hat[:42*522], y_hat[42*(564+horizon):]))

    num = np.sum((y_test - y_hat)**2)
    denom = np.sum(y_test**2)
    rsquared = 1 - num/denom
    return rsquared
