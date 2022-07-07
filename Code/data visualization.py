from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
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
amount = df.t.value_counts().sort_index()
plt.plot(SPX.Date, amount, '-', c='black', linewidth=0.5)
plt.xlabel("Date")
plt.ylabel("Number of options")
plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\options.png")

# Make 3D plot of one day
day = amount.argmax()
temp2 = df[df.t == day]
temp = temp2[temp2.daystoex >= 10]
temp = temp[temp.daystoex <= 252]
temp = temp[temp.moneyness >= 0.9]
temp = temp[temp.moneyness <= 1.3]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
ax.view_init(elev=14, azim=123)
# ax.set_title('surface')
ax.set_xlabel("Moneyness")
ax.set_ylabel("Tenor (days)")
ax.set_zlabel("Implied Volatility")
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\3D plot most not zoom.png")

# Make FULL 3D plot of a day
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(temp2.moneyness, temp2.daystoex, temp2.IV, cmap=cm.jet)
ax.view_init(elev=14, azim=123)
# ax.set_title('surface')
ax.set_xlabel("Moneyness")
ax.set_ylabel("Tenor (days)")
ax.set_zlabel("Implied Volatility")
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\3D plot most full not zoom.png")

# Make 3D plot of same day, but balanced panel
day = amount.argmax()
temp = df_bal[df_bal.t == day]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
ax.view_init(elev=14, azim=123)
# ax.set_title('surface')
ax.set_xlabel("Moneyness")
ax.set_ylabel("Tenor (days)")
ax.set_zlabel("Implied Volatility")
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\3D plot most balanced not zoom.png")

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
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\3D plot balanced not zoom.png")

# Plot the characteristics of the IVS over time: Level, skew, term structure.
# Level
level = np.zeros(SPX.shape[0])
for _ in range(len(level)):
    temp = df_bal[df_bal.t == _]
    level[_] = np.mean(temp.IV)

plt.figure()
plt.plot(SPX.Date, level, '-', c='black', linewidth=0.5)
plt.xlabel("Date")
plt.ylabel("IV level")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\IV level.png")

# Skew
skew = np.zeros(SPX.shape[0])
for _ in range(len(skew)):
    temp = df_bal[df_bal.t == _]
    high = temp[temp.moneyness == np.max(temp.moneyness)].IV.reset_index(drop=True)
    low = temp[temp.moneyness == np.min(temp.moneyness)].IV.reset_index(drop=True)
    skew[_] = np.mean(high - low)

plt.figure()
plt.plot(SPX.Date, skew, '-', c='black', linewidth=0.5)
plt.xlabel("Date")
plt.ylabel("IV skew")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\IV skew.png")

# Term Structure
term_structure = np.zeros(SPX.shape[0])
for _ in range(len(term_structure)):
    temp = df_bal[df_bal.t == _]
    long = temp[temp.daystoex == np.max(temp.daystoex)].IV.reset_index(drop=True)
    short = temp[temp.daystoex == np.min(temp.daystoex)].IV.reset_index(drop=True)
    term_structure[_] = np.mean(long - short)

plt.figure()
plt.plot(SPX.Date, term_structure, '-', c='black', linewidth=0.5)
plt.xlabel("Date")
plt.ylabel("IV term structure")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\IV term structure.png")

# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\IVS characteristics.png")