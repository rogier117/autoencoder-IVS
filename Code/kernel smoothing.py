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
from matplotlib import cm

df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data IV OTM.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')

tenors = [10, 21, 63, 126, 189, 252]
moneyness = [0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3]
gridsize = len(tenors) * len(moneyness)

dates = df.date.unique()
t = df.t.unique()

grid = np.meshgrid(tenors, moneyness)
daystoex = np.tile(grid[0].flatten(), len(t))
moneyness = np.tile(grid[1].flatten(), len(t))

dates = np.repeat(dates, gridsize)
t = np.repeat(t, gridsize)
# Balanced dataframe df_bal
df_bal = pd.DataFrame()
df_bal['Date'] = dates
df_bal['t'] = t
df_bal['daystoex'] = daystoex
df_bal['moneyness'] = moneyness
# 'normalize' tenor and moneyness for equal importance in the kernel smoother
df_bal['tau_nor'] = (daystoex - 10) / 242
df_bal['k_nor'] = (moneyness - 0.9) / 0.4

IV = np.zeros(df_bal.shape[0])
# b is the smoothing parameter
b = 5/100
for _ in tqdm(range(df_bal.shape[0]), desc='option'):
    if _ == 0 or not df_bal.t[_] == df_bal.t[_ - 1]:
        temp = df[df.t == df_bal.t[_]].reset_index(drop=True)
        temp['tau_nor'] = (temp.daystoex - 10) / 242
        temp['k_nor'] = (temp.moneyness - 0.9) / 0.4
    tau_temp = np.repeat(df_bal.tau_nor[_], temp.shape[0])
    k_temp = np.repeat(df_bal.k_nor[_], temp.shape[0])

    euc_sq = (tau_temp - temp.tau_nor) ** 2 + (k_temp - temp.k_nor) ** 2
    # kernel is the weight that each point gets
    kernel = np.exp(- euc_sq / (b ** 2))
    IV[_] = np.dot(temp.IV, kernel) / np.sum(kernel)

df_bal['IV'] = IV

df_bal.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv', index=False)

