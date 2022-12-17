import pandas as pd
import numpy as np


df = pd.read_csv('D:/Datasets/Mall_Customers.csv')
print(df.head())

X = df.iloc[:,[3,4]].values

from sklearn.cluster import DBSCAN
model = DBSCAN(eps=3, min_samples=4)
model.fit(X)

label = model.labels_
print(label)