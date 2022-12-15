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

df = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data unbalanced.csv')
df_bal = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\option data balanced.csv')
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
r = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\riskfree rate data cleaned.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
df_bal['Date'] = pd.to_datetime(df_bal['Date'], format='%Y-%m-%d')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
r['DATE'] = pd.to_datetime(r['DATE'], format='%Y-%m-%d')

day = [0,1000,2000,3000,4000,5000]
for i in range(len(day)):
    temp2 = df[df.t == day[i]]

    temp = temp2[temp2.daystoex >= 10]
    temp = temp[temp.daystoex <= 252]
    temp = temp[temp.moneyness >= 0.9]
    temp = temp[temp.moneyness <= 1.3]


    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
    ax.view_init(elev=90, azim=90)
    # ax.set_title('surface')
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Tenor (days)")
    ax.set_zlabel("Implied Volatility")
    plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\Powerpoint\top view unbalanced " + str(i) + ".png")

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
    ax.view_init(elev=14, azim=123)
    # ax.set_title('surface')
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Tenor (days)")
    ax.set_zlabel("Implied Volatility")
    plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\Powerpoint\normal view unbalanced " + str(i) + ".png")

    temp = df_bal[df_bal.t == day[i]]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
    ax.view_init(elev=90, azim=90)
    # ax.set_title('surface')
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Tenor (days)")
    ax.set_zlabel("Implied Volatility")
    plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\Powerpoint\top view balanced " + str(i) + ".png")

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(temp.moneyness, temp.daystoex, temp.IV, cmap=cm.jet)
    ax.view_init(elev=14, azim=123)
    # ax.set_title('surface')
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Tenor (days)")
    ax.set_zlabel("Implied Volatility")
    plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\Powerpoint\normal view balanced " + str(i) + ".png")