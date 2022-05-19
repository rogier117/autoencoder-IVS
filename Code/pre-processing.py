import pandas as pd
import numpy as np
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y%m%d')
df['daystoex'] = df['exdate']-df['date']  # Might need to be changed to trading days only
