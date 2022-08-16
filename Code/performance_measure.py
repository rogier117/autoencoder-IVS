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
