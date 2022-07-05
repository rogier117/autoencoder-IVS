from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
from tqdm import tqdm

df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data IV OTM.csv')
df_bal = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
df_bal['Date'] = pd.to_datetime(df_bal['Date'], format='%Y-%m-%d')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')

# Amount of different options per day
amount=df.t.value_counts().sort_index()
plt.plot(SPX.Date,amount,'-',c='black',linewidth=0.5)
plt.xlabel("Date")
plt.ylabel("Number of options")
plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\options.png")

# Make 3D plot of one day
day = amount.argmax()
temp = df[df.t == day]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
ax.view_init(elev=14, azim=123)
# ax.set_title('surface')
ax.set_xlabel("Moneyness")
ax.set_ylabel("Tenor (days)")
ax.set_zlabel("Implied Volatility")
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\3D plot.png")

# Make 3D plot of average IV for the balanced panel
temp = pd.DataFrame()
temp['moneyness'] = df_bal[df_bal.t == 0].moneyness
temp['daystoex'] = df_bal[df_bal.t == 0].daystoex
IV = np.zeros(temp.shape[0])
for _ in range(temp.shape[0]):
    temp2 = df_bal[df_bal.moneyness == temp.moneyness[_]]
    temp2 = temp2[temp2.daystoex == temp.daystoex[_]]
    IV[_] = np.mean(temp2.IV)
temp['IV'] = IV

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
ax.view_init(elev=12, azim=114)
# ax.set_title('surface of balanced panel')
ax.set_xlabel("Moneyness")
ax.set_ylabel("Tenor (days)")
ax.set_zlabel("Implied Volatility")
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\3D plot balanced.png")