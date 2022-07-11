import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# importing or loading the dataset
dataset = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\X balanced.csv')

# distributing the dataset into two components X and Y
X = dataset.values
X_train, X_test = train_test_split(X, test_size = 0.20, shuffle=False)

# performing preprocessing part
sc = StandardScaler()
X_train_nor = sc.fit_transform(X_train)
X_test_nor = sc.transform(X_test)

# applying PCA
pca = PCA(n_components=3)
f_train = pca.fit_transform(X_train_nor)
f_test = pca.transform(X_test_nor)

# Theoretical explained variance in the training sample
explained_variance = pca.explained_variance_ratio_

X_hat_nor = pca.inverse_transform(f_test)
X_hat = sc.inverse_transform(X_hat_nor)

# Calculate total R^2 as described in paper
numerator = np.sum((X_test_nor - X_hat_nor) ** 2)
denominator = np.sum((X_test_nor - np.mean(X_test_nor)) ** 2) # CHANGE THE MEAN TO MEAN PER COLUMN

total_Rsq = 1 - numerator/denominator

# Calculate daily total R^2
Rsq_array = np.zeros(X_hat_nor.shape[0])
for i in range(X_hat_nor.shape[0]):
    temp_hat = X_hat_nor[i,:]
    temp_test = X_test_nor[i,:]
    temp_num = np.sum((temp_test - temp_hat) ** 2)
    temp_denom = np.sum(temp_test ** 2)
    Rsq_array[i] = 1 - temp_num / temp_denom

# Plotting with dates: import dataset with dates
SPX = pd.read_csv(r'D:\Master Thesis\autoencoder-IVS\Data\SPX data date.csv')
SPX['Date'] = pd.to_datetime(SPX['Date'], format='%Y-%m-%d')
plt.plot(SPX.Date[-len(Rsq_array):],Rsq_array)
