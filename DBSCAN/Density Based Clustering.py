import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pandas.read_csv("cc_general.csv")
#print(data)
X = data.drop("CUST_ID", axis='columns')
X.fillna(method='ffill', inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(eps=0.0375,min_samples=3)
y_predicted = model.fit_predict(X_scaled)
data['clusters'] = y_predicted

print(data.clusters.unique())