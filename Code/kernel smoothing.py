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

df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data IV OTM.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')

tenors = [10, 21, 63, 126, 189, 252]
moneyness = [0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3]

dates = df.date.unique()