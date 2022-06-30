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

# # Import raw option data
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data.csv')
# df = df.drop(columns=['issuer', 'exercise_style', 'open_interest', 'impl_volatility', 'delta', 'gamma', 'vega',
#                       'theta', 'optionid', 'contract_size', 'forward_price', 'index_flag'])
#
# # Remove 'volume = 0' entries
# df = df[df.volume != 0]
# df = df.reset_index(drop=True)
#
# # CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data volume.csv', index=False)
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data volume.csv')
# # CHECKPOINT END
#
# # Import raw index data
# SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data.csv', thousands=',')
#
# # Change the dates from strings to actual datetimes
# df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
# df['exdate'] = pd.to_datetime(df['exdate'], format='%Y%m%d')
# SPX['Date'] = pd.to_datetime(SPX['Date'], format='%m/%d/%Y')
#
# # Count (business) days to expiry (+- 2 hours)
# # df['daystoex'] = np.busday_count(df['date'].values.astype('datetime64[D]'),df['exdate'].values.astype('datetime64[D]'))
# nyse = mcal.get_calendar('NYSE')
# daystoex = np.zeros(df.shape[0])
# for _ in tqdm(range(df.shape[0]),desc='_'):
#     daystoex[_] = len(nyse.valid_days(start_date=df['date'][_],end_date=df['exdate'][_]) - 1
# df['daystoex'] = daystoex
# # Only keep if 5<=daystoex<=300
# df = df[df.daystoex >= 5].reset_index(drop=True)
# df = df[df.daystoex <= 300].reset_index(drop=True)
#
# # Check whether the option dates and stock dates are identical
# if np.sum(SPX.Date == df.date.unique()) != np.max([SPX.shape[0],df.date.unique().shape[0]]):
#     raise ValueError('Option and stock dates do not match.')
#
# # Once we have checked that the dates correspond, we can give the dates numbers (duration +- 3 mins)
# t = np.zeros(df.shape[0]).astype(int)
# count = 0
# for _ in range(df.shape[0]):
#     if _ == 0:
#         t[_] = count
#     elif df['date'][_] == df['date'][_-1]:
#         t[_] = count
#     else:
#         count += 1
#         t[_] = count
# df['t'] = t
#
# # Calculate moneyness as: stock price (close) / strike price
# df['moneyness'] = SPX['Close'][df['t']].reset_index(drop=True) / (df['strike_price']/1000)
# # Only keep if 0.8<=moneyness<=1.5
# df = df[df.moneyness >= 0.8].reset_index(drop=True)
# df = df[df.moneyness <= 1.5].reset_index(drop=True)
#
# # CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data volume moneyness.csv', index=False)
# SPX.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv', index=False)
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data volume moneyness.csv')
# SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
# SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
# # CHECKPOINT END
#
# # Riskfree rate
# r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data.csv')
# r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')
# r = r[r.DATE.isin(SPX.Date)].reset_index(drop=True)
# r['DTB3'] = pd.to_numeric(r['DTB3'], errors='coerce')
# r['DTB3'] = r['DTB3'].interpolate(method='linear', axis=0)
#
# # Empirical dividend yield
# emp_q = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\dividend yield data.csv')
# emp_q = emp_q.drop(columns=['secid'])
# emp_q['date'] = pd.to_datetime(emp_q['date'], format='%Y%m%d')
#
#
# # Midpoint option price
# df['price'] = (df['best_bid'] + df['best_offer']) / 2
#
# # Remove options with price lower than 1/8
# df = df[df.price >= 0.125].reset_index(drop=True)
#
# # Remove unnecessary columns
# df = df.drop(columns=['issuer', 'exercise_style', 'open_interest', 'impl_volatility', 'delta', 'gamma', 'vega', 'theta',
#                       'optionid', 'contract_size', 'forward_price', 'index_flag'])
#
# # Find put-call pairs as closest to ATM for every time-to-maturity at each point in time (+- 2:10 mins)
# df['moneynessdev'] = (df['moneyness'] - 1) ** 2
# qvector = np.zeros(df.shape[0])
# begintime = time.time()
# for _ in tqdm(df.t.unique(), desc='day'):
#     temp = df[df.t == _]
#     for tenor in temp.daystoex.unique():
#         temp2 = temp[temp.daystoex == tenor]
#         temp2 = temp2.sort_values(by='moneynessdev')
#         finished = False
#         i = 0
#         while not finished:
#             if i == temp2.shape[0]-1:
#                 finished = True
#                 q = math.nan
#                 qvector[temp2.index.values] = q
#             elif temp2.moneyness.iloc[i] == temp2.moneyness.iloc[i+1] and temp2.cp_flag.iloc[i] != temp2.cp_flag.iloc[i+1]:
#                 finished = True
#                 if temp2.cp_flag.iloc[i] == 'C':
#                     call = temp2.iloc[i]
#                     put = temp2.iloc[i+1]
#                 else:
#                     call = temp2.iloc[i+1]
#                     put = temp2.iloc[i]
#                 q = - (1 / (tenor/252)) * math.log((call.price - put.price + (call.strike_price/1000) * math.exp(-(r.DTB3[_]/100) * (tenor/252))) / SPX.Close[_])
#                 qvector[temp2.index.values] = q
#             else:
#                 i += 1
# df['q'] = qvector
#
# # Replace NaN values with empirical dividend yields
# for _ in df.q.isna().to_numpy().nonzero()[0]:
#     df.at[_, 'q'] = emp_q.loc[df.t[_], 'rate']/100
# total_time = time.time()-begintime
#
# # CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data dividend yield.csv', index=False)
# r.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv', index=False)
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data dividend yield.csv')
# SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
# r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
# SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
# r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')
# # CHECKPOINT END
#
# # No arbitrage conditions (+- 8:30 min)
# arbitrage = np.full(df.shape[0],False)
# for _ in tqdm(range(df.shape[0]), desc='option'):
#     if df.cp_flag[_] == 'C':
#         if df.price[_] < SPX.Close[df.t[_]] * math.exp(- df.q[_] * (df.daystoex[_]/252)) - (df.strike_price[_]/1000) * math.exp(- (r.DTB3[df.t[_]]/100) * (df.daystoex[_]/252)):
#             arbitrage[_] = True
#     else:
#         if df.price[_] < (df.strike_price[_]/1000) * math.exp(- (r.DTB3[df.t[_]]/100) * (df.daystoex[_]/252)) - SPX.Close[df.t[_]] * math.exp(- df.q[_] * (df.daystoex[_]/252)):
#             arbitrage[_] = True
# arbitrage = arbitrage.nonzero()[0]
# df = df.drop(index=arbitrage).reset_index(drop=True)
#
#
# # CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data arbitrage free.csv', index=False)
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data arbitrage free.csv')
# SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
# r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
# SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
# r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')
# # CHECKPOINT END
#
# # DEBUGGING
# np.seterr(all='raise')
#
# # Calculate Implied Volatilities using Newton Raphson algorithm
# df.sort_values(by=['t','strike_price','daystoex','cp_flag'], inplace=True)
# iv = np.zeros(df.shape[0])
# for _ in tqdm(range(df.shape[0]), desc='option'):
#     if not _ == 0 and df.t.iloc[_] == df.t.iloc[_-1] and df.strike_price.iloc[_] == df.strike_price.iloc[_-1] and df.daystoex.iloc[_] == df.daystoex.iloc[_-1]:
#         ivtemp = iv[_-1]
#     else:
#         if df.cp_flag.iloc[_] == 'P':
#             call_price = df.price.iloc[_] + SPX.Close.iloc[df.t.iloc[_]] * math.exp(- df.q.iloc[_] * (df.daystoex.iloc[_]/252)) - (df.strike_price.iloc[_]/1000) * math.exp(- (r.DTB3.iloc[df.t.iloc[_]]/100) * (df.daystoex.iloc[_]/252))
#         else:
#             call_price = df.price.iloc[_]
#         ivtemp = np.maximum(math.sqrt((2 * math.pi)/(df.daystoex.iloc[_]/252)) * (call_price/SPX.Close.iloc[df.t.iloc[_]]), 0.1)
#         it = 0
#         d1 = (math.log(SPX.Close.iloc[df.t.iloc[_]] / (df.strike_price.iloc[_]/1000)) + ((r.DTB3.iloc[df.t.iloc[_]]/100) - df.q.iloc[_] + ivtemp ** 2 / 2) * (df.daystoex.iloc[_]/252)) / (ivtemp * np.sqrt((df.daystoex.iloc[_]/252)))
#         d2 = d1 - ivtemp * np.sqrt((df.daystoex.iloc[_]/252))
#         call = SPX.Close.iloc[df.t.iloc[_]] * np.exp(-df.q.iloc[_] * (df.daystoex.iloc[_]/252)) * norm.cdf(d1) - norm.cdf(d2) * (df.strike_price.iloc[_]/1000) * np.exp(-(r.DTB3.iloc[df.t.iloc[_]]/100) * (df.daystoex.iloc[_]/252))
#         diff = call - call_price
#         while it < 100 and abs(diff) > 0.0001:
#             vega = np.exp(-df.q.iloc[_] * (df.daystoex.iloc[_]/252)) * SPX.Close.iloc[df.t.iloc[_]] * norm.pdf(d1) * np.sqrt((df.daystoex.iloc[_]/252))
#             ivtemp = ivtemp - np.clip(diff / vega, -0.1, 0.1)
#             d1 = (math.log(SPX.Close.iloc[df.t.iloc[_]] / (df.strike_price.iloc[_] / 1000)) + (
#                         (r.DTB3.iloc[df.t.iloc[_]] / 100) - df.q.iloc[_] + ivtemp ** 2 / 2) * (
#                               df.daystoex.iloc[_] / 252)) / (ivtemp * np.sqrt((df.daystoex.iloc[_] / 252)))
#             d2 = d1 - ivtemp * np.sqrt((df.daystoex.iloc[_] / 252))
#             call = SPX.Close.iloc[df.t.iloc[_]] * np.exp(-df.q.iloc[_] * (df.daystoex.iloc[_] / 252)) * norm.cdf(
#                 d1) - norm.cdf(d2) * (df.strike_price.iloc[_] / 1000) * np.exp(
#                 -(r.DTB3.iloc[df.t.iloc[_]] / 100) * (df.daystoex.iloc[_] / 252))
#             diff = call - call_price
#             it += 1
#     iv[_] = ivtemp
# df['IV'] = iv
# df.sort_index()
#
# # CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data IV.csv', index=False)
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data IV.csv')
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')
# CHECKPOINT END

# Transform all options to OTM options: moneyness<=1 ==> Call, moneyness>1 ==> Put (+- 8 minutes)
prices = np.empty(df.shape[0])
cp = []
changed = np.full(df.shape[0],False)
for _ in tqdm(range(df.shape[0]), desc='option'):
    if df.moneyness.iloc[_] <= 1 and df.cp_flag.iloc[_] == 'P':
        # Change put to call
        prices[_] = df.price.iloc[_] + SPX.Close.iloc[df.t.iloc[_]] * math.exp(
            - df.q.iloc[_] * (df.daystoex.iloc[_] / 252)) - (
                            df.strike_price.iloc[_] / 1000) * math.exp(
            - (r.DTB3.iloc[df.t.iloc[_]] / 100) * (df.daystoex.iloc[_] / 252))
        cp.append('C')
        changed[_] = True

    elif df.moneyness[_] > 1 and df.cp_flag[_] == 'C':
        # Change call to put
        prices[_] = df.price.iloc[_] - SPX.Close.iloc[df.t.iloc[_]] * math.exp(
            - df.q.iloc[_] * (df.daystoex.iloc[_] / 252)) + (
                            df.strike_price.iloc[_] / 1000) * math.exp(
            - (r.DTB3.iloc[df.t.iloc[_]] / 100) * (df.daystoex.iloc[_] / 252))
        cp.append('P')
        changed[_] = True
    else:
        # Keep it as it is
        prices[_] = df.price.iloc[_]
        cp.append(df.cp_flag[_])

df['price'] = prices
df['cp_flag'] = cp

# # CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data IV OTM.csv', index=False)
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data IV OTM.csv')
# SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
# r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
# df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
# SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
# r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')
# # CHECKPOINT END

# plot days to expiry histogram
# plt.hist(df['daystoex'], bins=50)
# plt.show()
