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

# SPX_data.to_csv(path_or_buf=r'D:\Master Thesis\autoencoder-IVS\Data\SPX data.csv')
# SPX_data = pd.concat([data02,data03,data04,data05,data06,data07,data08,data09,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21])

# data02 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2002.csv')
# data03 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2003.csv')
# data04 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2004.csv')
# data05 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2005.csv')
# data06 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2006.csv')
# data07 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2007.csv')
# data08 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2008.csv')
# data09 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2009.csv')
# data10 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2010.csv')
# data11 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2011.csv')
# data12 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2012.csv')
# data13 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2013.csv')
# data14 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2014.csv')
# data15 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2015.csv')
# data16 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2016.csv')
# data17 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2017.csv')
# data18 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2018.csv')
# data19 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2019.csv')
# data20 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2020.csv')
# data21 = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX los\SPX 2021.csv')
#
# data02 = data02.iloc[::-1]
# data03 = data03.iloc[::-1]
# data04 = data04.iloc[::-1]
# data05 = data05.iloc[::-1]
# data06 = data06.iloc[::-1]
# data07 = data07.iloc[::-1]
# data08 = data08.iloc[::-1]
# data09 = data09.iloc[::-1]
# data10 = data10.iloc[::-1]
# data11 = data11.iloc[::-1]
# data12 = data12.iloc[::-1]
# data13 = data13.iloc[::-1]
# data14 = data14.iloc[::-1]
# data15 = data15.iloc[::-1]
# data16 = data16.iloc[::-1]
# data17 = data17.iloc[::-1]
# data18 = data18.iloc[::-1]
# data19 = data19.iloc[::-1]
# data20 = data20.iloc[::-1]
# data21 = data21.iloc[::-1]