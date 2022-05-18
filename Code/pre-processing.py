import pandas as pd
import numpy as np
df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data.csv')
df = df.to_numpy
print(df[0:10,0:10])