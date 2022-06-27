import math
import time
from scipy.stats import norm

import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_market_calendars as mcal #CHANGE ALL USFEDERALHOLIDAY THINGS WITH THIS ONE!!

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

# Import covariates

# Function to let all dates correspond to SPX
def match_dates(good, good_colname, new, new_colname):
    # us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # first_date = pd.date_range(end=good[good_colname][0], periods=22, freq=us_bd)
    nyse = mcal.get_calendar('NYSE')
    first_date = nyse.valid_days(start_date=good[good_colname][0] - timedelta(days=40), end_date=good[good_colname][0])[-22]
    pre_sample = nyse.valid_days(start_date=first_date, end_date=good[good_colname][0])
    new = new[new[new_colname] >= pre_sample[0]].reset_index(drop=True)
    keep = np.full(new.shape[0], False)
    for _ in range(new.shape[0]):
        if new[new_colname][_] in pre_sample or new[new_colname][_] in good[good_colname].unique():
            keep[_] = True
    new = new[keep].reset_index(drop=True)
    check = good[good_colname].isin(new[new_colname])
    check = np.array([not elem for elem in check]).nonzero()[0]
    if new.shape[0] < good.shape[0] + 21:
        positions = np.arange(new.shape[0], good.shape[0] + 21)
        for _ in range(good.shape[0] + 21 - new.shape[0]):
            new.loc[positions[_]] = float("nan")
            new.at[positions[_], new_colname] = good[good_colname][check[_]]
    new = new.sort_values(by=new_colname, ignore_index=True)
    return new


# Volatility Index (VIX)
VIX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\VIX_History.csv')
VIX['DATE'] = pd.to_datetime(VIX['DATE'], format='%m/%d/%Y')
VIX = match_dates(good=SPX, good_colname='Date', new=VIX, new_colname='DATE')
VIX['CLOSE'] = VIX['CLOSE'].interpolate(method='linear', axis=0)

# Left Tail Volatility (LTV) (ONLY GOES TO END OF 2019!!)
LTV = pd.read_html(r'D:\Master Thesis\autoencoder-IVS\Data\LTV.xls', header=0, decimal=',', thousands='.')
LTV = LTV[0]
LTV['Date'] = pd.to_datetime(LTV['Date'], format='%Y-%m-%d')
LTV = match_dates(good=SPX, good_colname='Date', new=LTV, new_colname='Date')
LTV['Index'] = LTV['Index'].interpolate(method='linear', axis=0)

# Left Tail Porbability (LTP) (ONLY GOES TO END OF 2019!!)
LTP = pd.read_html(r'D:\Master Thesis\autoencoder-IVS\Data\LTP.xls', header=0, decimal=',', thousands='.')
LTP = LTP[0]
LTP['Date'] = pd.to_datetime(LTP['Date'], format='%Y-%m-%d')
LTP = match_dates(good=SPX, good_colname='Date', new=LTP, new_colname='Date')
LTP['Index'] = LTP['Index'].interpolate(method='linear', axis=0)

# Realized Volatility (RVOL)
RVOL = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\RVOL.csv')
RVOL = RVOL.rename(columns={'Unnamed: 0':'Date'})
RVOL = RVOL[RVOL['Symbol'] == '.SPX'].reset_index(drop=True)
RVOL = RVOL[['Date', 'rv5']]
# Remove Timezone as it is of no importance
for _ in range(RVOL.shape[0]):
    RVOL.at[_, 'Date'] = RVOL.Date.iloc[_][:-6]
RVOL['Date'] = pd.to_datetime(RVOL['Date'], format="%Y-%m-%d")
RVOL = match_dates(good=SPX, good_colname='Date', new=RVOL, new_colname='Date')
RVOL['rv5'] = RVOL['rv5'].interpolate(method='linear', axis=0)

# Economic Policy Uncertainty (EPU) (monthly)
EPU = pd.read_excel(r'D:\Master Thesis\autoencoder-IVS\Data\EPU.xlsx')
EPU = EPU.rename(columns={'Three_Component_Index': 'EPU'})
EPU = EPU[:-1]
EPU['Month'] = EPU['Month'].shift(1)
EPU['Year'] = EPU['Year'].shift(1)
EPU = EPU.iloc[1: , :]
EPU = EPU.astype({'Month': 'int32'})
EPU['Year'] = EPU.Year.astype(str)
EPU['Month'] = EPU.Month.astype(str)
EPU['Date'] = EPU.Year + '-' + EPU.Month + '-1'
EPU['Date'] = pd.to_datetime(EPU['Date'], format='%Y-%m-%d')
EPU = EPU[['Date','EPU']].reset_index(drop=True)
nyse = mcal.get_calendar('NYSE')
dates = nyse.valid_days(start_date=EPU.Date[0],end_date=EPU.Date[EPU.shape[0]-1])
dates = list(set(dates) - set(EPU.Date))
for _ in range(len(dates)):
    el = EPU.shape[0]
    EPU.loc[el] = float("nan")
    EPU.at[el, 'Date'] = dates[_]
EPU = EPU.sort_values(by='Date').reset_index(drop=True)
EPU = EPU.fillna(method='ffill')
EPU = match_dates(good=SPX, good_colname='Date', new=EPU, new_colname='Date')

# US News Index (USNI)


# plot days to expiry histogram
# plt.hist(df['daystoex'], bins=50)
# plt.show()
