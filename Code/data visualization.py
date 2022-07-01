from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats

df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data IV OTM.csv')
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')

# Amount of different options per day
amount=df.t.value_counts().sort_index()
plt.plot(SPX.Date,amount,'-',c='black',linewidth=0.5)
plt.xlabel("Date")
plt.ylabel("Number of options")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\options.png")

# Make 3D plot of one day
day = amount.argmax()
temp = df[df.t == day]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
ax.view_init(elev=14, azim=123)
ax.set_title('surface')
ax.set_xlabel("Moneyness")
ax.set_ylabel("Tenor (days)")
ax.set_zlabel("Implied Volatility")
plt.savefig("3D plot.png")
