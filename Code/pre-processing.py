import math
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# # Import raw option data
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data.csv')
# df = df.drop(columns=['issuer', 'exercise_style', 'open_interest', 'impl_volatility', 'delta', 'gamma', 'vega',
#                       'theta', 'optionid', 'contract_size', 'forward_price', 'index_flag'])
#
# # Remove volume = 0 entries
# df = df[df.volume != 0]
# df = df.reset_index(drop=True)

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
# # Count (business) days to expiry
# df['daystoex'] = np.busday_count(df['date'].values.astype('datetime64[D]'),df['exdate'].values.astype('datetime64[D]'))
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
# df = df[df.price >= 0.125]
# df = df.reset_index(drop=True)
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
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data dividend yield.csv')
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')
# CHECKPOINT END

# No arbitrage conditions (+- 8:30 min)
arbitrage = np.full(df.shape[0],False)
for _ in tqdm(range(df.shape[0]), desc='option'):
    if df.cp_flag[_] == 'C':
        if df.price[_] < SPX.Close[df.t[_]] * math.exp(- df.q[_] * (df.daystoex[_]/252)) - (df.strike_price[_]/1000) * math.exp(- (r.DTB3[df.t[_]]/100) * (df.daystoex[_]/252)):
            arbitrage[_] = True
    else:
        if df.price[_] < (df.strike_price[_]/1000) * math.exp(- (r.DTB3[df.t[_]]/100) * (df.daystoex[_]/252)) - SPX.Close[df.t[_]] * math.exp(- df.q[_] * (df.daystoex[_]/252)):
            arbitrage[_] = True
arbitrage = arbitrage.nonzero()[0]
df = df.drop(index=arbitrage).reset_index(drop=True)


# plot days to expiry histogram
# plt.hist(df['daystoex'], bins=50)
# plt.show()
