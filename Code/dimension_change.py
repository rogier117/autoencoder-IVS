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

df_bal = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')
df_bal['Date'] = pd.to_datetime(df_bal['Date'], format='%Y-%m-%d')

t = df_bal.t.unique()
gridsize = df_bal[df_bal.t == 0].shape[0]
# Convert into processable dataframe
X_mat = np.zeros((len(t), gridsize))
j = 0
for _ in tqdm(range(df_bal.shape[0]), desc='_'):
    if not _ == 0 and not df_bal.t[_] == df_bal.t[_-1]:
        j = 0
    X_mat[df_bal.t[_], j] = df_bal.IV[_]
    j += 1

X = pd.DataFrame(X_mat)
X.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv', index=False)
