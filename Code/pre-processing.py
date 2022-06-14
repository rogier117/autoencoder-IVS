import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Import raw option data
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data.csv')
#
# # Remove volume = 0 entries
# df = df[df.volume != 0]

# CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data volume.csv', index=False)
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data volume.csv')
# CHECKPOINT END

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
# # Get mid prices
# SPX['Mid'] = (SPX['High'] + SPX['Low'])/2
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
# # Calculate moneyness as: stock price (mid) / strike price
# df['moneyness'] = SPX['Mid'][df['t']].reset_index(drop=True) / (df['strike_price']/1000)

# CHECKPOINT BEGIN
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data volume moneyness.csv', index=False)
# SPX.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\SPX data mid.csv', index=False)
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data volume moneyness.csv')
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data mid.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
# CHECKPOINT END

# Riskfree rate
r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data.csv')
r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')
r = r[r.DATE.isin(SPX.Date)].reset_index(drop=True)

# Midpoint option price
df['price'] = (df['best_bid'] + df['best_offer'])/2

# Remove options with price lower than 1/8
df = df[df.price >= 0.125]

# Remove unnecessary columns (ADD MORE COLUMNS TOMORROW)
df = df.drop(columns=['issuer', 'exercise_style'])







# plot days to expiry histogram
# plt.hist(df['daystoex'], bins=50)
# plt.show()