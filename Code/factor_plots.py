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

df = pd.read_excel(r"D:\Master Thesis\Results\factor plot.xlsx")

# PCA
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df.level, '-', linewidth=0.8, label="Level")
# ax.plot(df.Date, df["PCA-1"], '-', linewidth=0.8, label="PCA-1")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# # plt.show()
# # plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_PCA-1.png")
#
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df.term_structure, '-', linewidth=0.8, label="Term structure")
# ax.plot(df.Date, df["PCA-2"], '-', linewidth=0.8, label="PCA-2")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# # plt.show()
# # plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_PCA-2.png")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df["skew"], '-', linewidth=0.8, label="Skew")
# ax.plot(df.Date, df["PCA-3"], '-', linewidth=0.8, label="PCA-3")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# # plt.show()
# # plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_PCA-3.png")


# # IPCA_B
# cmap = plt.get_cmap("tab10")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df["IPCA_B-1"], '-', linewidth=0.8, color=cmap(1), label="IPCA$_B$-1")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_IPCA_B-1.png")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df.term_structure, '-', linewidth=0.8, label="Term structure")
# ax.plot(df.Date, df["IPCA_B-2"], '-', linewidth=0.8, label="IPCA$_B$-2")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_IPCA_B-2.png")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df.term_structure, '-', linewidth=0.8, label="Term structure")
# ax.plot(df.Date, df["IPCA_B-3"], '-', linewidth=0.8, label="IPCA$_B$-3")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_IPCA_B-3.png")


# # IPCA_U
# cmap = plt.get_cmap("tab10")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df["IPCA_U-1"], '-', linewidth=0.8, color=cmap(1), label="IPCA$_U$-1")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_IPCA_U-1.png")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df.term_structure, '-', linewidth=0.8, label="Term structure")
# ax.plot(df.Date, df["IPCA_U-2"], '-', linewidth=0.8, label="IPCA$_U$-2")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_IPCA_U-2.png")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df["skew"], '-', linewidth=0.8, label="Skew")
# ax.plot(df.Date, df["IPCA_U-3"], '-', linewidth=0.8, label="IPCA$_U$-3")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_IPCA_U-3.png")


# # AE_1
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df.level, '-', linewidth=0.8, label="Level")
# ax.plot(df.Date, df["AE_1-1"], '-', linewidth=0.8, label="AE$_1$-1")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_AE_1-1.png")
#
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df.term_structure, '-', linewidth=0.8, label="Term structure")
# ax.plot(df.Date, df["AE_1-2"], '-', linewidth=0.8, label="AE$_1$-2")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_AE_1-2.png")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(df.Date, df["skew"], '-', linewidth=0.8, label="Skew")
# ax.plot(df.Date, df["AE_1-3"], '-', linewidth=0.8, label="AE$_1$-3")
# plt.legend(loc="upper left", frameon=False)
# plt.xlabel("Date")
# plt.ylabel("")
# plt.show()
# plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_AE_1-3.png")



# AE_2

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df.Date, df.level, '-', linewidth=0.8, label="Level")
ax.plot(df.Date, df["AE_2-1"], '-', linewidth=0.8, label="AE$_2$-1")
plt.legend(loc="upper left", frameon=False)
plt.xlabel("Date")
plt.ylabel("")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_AE_2-1.png")


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df.Date, df.level, '-', linewidth=0.8, label="Level")
ax.plot(df.Date, df["AE_2-2"], '-', linewidth=0.8, label="AE$_2$-2")
plt.legend(loc="upper left", frameon=False)
plt.xlabel("Date")
plt.ylabel("")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_AE_2-2.png")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df.Date, df["skew"], '-', linewidth=0.8, label="Skew")
ax.plot(df.Date, df["AE_2-3"], '-', linewidth=0.8, label="AE$_2$-3")
plt.legend(loc="upper left", frameon=False)
plt.xlabel("Date")
plt.ylabel("")
plt.show()
plt.savefig(r"D:\Master Thesis\autoencoder-IVS\Figures\factor_plot_AE_2-3.png")
