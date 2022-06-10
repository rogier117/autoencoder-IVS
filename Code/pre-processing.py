import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y%m%d')
# df['daystoex'] = df['exdate']-df['date']  # Total days
df['daystoex'] = np.busday_count(df['date'].values.astype('datetime64[D]'),df['exdate'].values.astype('datetime64[D]')) # Business days

# plot days to expiry histogram
# plt.hist(df['daystoex'], bins=15)
# plt.show()