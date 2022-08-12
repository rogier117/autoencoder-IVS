import numpy as np


def rsq(y_test, y_hat):
    y_test = y_test.flatten()
    y_hat = y_hat.flatten()

    num = np.sum((y_test - y_hat)**2)
    denom = np.sum(y_test**2)
    rsquared = 1 - num/denom
    return rsquared
