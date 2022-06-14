import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Import raw option data
# df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data.csv')
#
# # Remove volume = 0 entries
# df = df[df.volume != 0]

# CHECKPOINT
# df.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\option data volume.csv')
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data volume.csv')

# Import raw index data
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data.csv', thousands=',')

# Change the dates from strings to actual datetimes
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y%m%d')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%m/%d/%Y')

# Count (business) days to expiry
df['daystoex'] = np.busday_count(df['date'].values.astype('datetime64[D]'),df['exdate'].values.astype('datetime64[D]'))

# Check whether the option dates and stock dates are identical
if np.sum(SPX.Date == df.date.unique()) != np.max([SPX.shape[0],df.date.unique().shape[0]]):
    raise ValueError('Option and stock dates do not match.')

# Get mid prices
SPX['Mid'] = (SPX['High'] + SPX['Low'])/2










# plot days to expiry histogram
# plt.hist(df['daystoex'], bins=50)
# plt.show()