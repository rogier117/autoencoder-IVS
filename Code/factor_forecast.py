from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import cm
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from matplotlib.pyplot import figure

# 522-563

df = pd.read_excel(r"D:\Master Thesis\Results\factor forecast.xlsx")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df.Date, df.PCA_normalized, '-', linewidth=0.8, label="PCA")
ax.plot(df.Date, df.AE_1_normalized, '-', linewidth=0.8, label="AE$_{1,I}$")
ax.axvspan(df.Date.iloc[522], df.Date.iloc[563], alpha=0.4, color='grey', label="Recession")
plt.legend(loc="upper right", frameon=False)
plt.xlabel("Date")
plt.ylabel("Absolute forecasting error")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_forecast.png")
