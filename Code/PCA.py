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
X_train, X_test = train_test_split(X, test_size = 0.15, shuffle=False)

# performing preprocessing part
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# applying PCA
pca = PCA(n_components=3)
f_train = pca.fit_transform(X_train)
f_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
