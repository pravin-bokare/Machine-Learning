import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

# Visualisation

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = scaler.fit_transform(df)

from sklearn.decomposition import PCA
model = PCA(n_components=2)
model.fit(df)

x_pca = model.transform(df)
print(x_pca.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

plt.show()
