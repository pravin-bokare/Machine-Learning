import pandas as pd

df = pd.read_csv("D:/Datasets/student_clustering.csv")

import matplotlib.pyplot as plt
#plt.scatter(df['cgpa'], df['iq'])

#Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,20):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    wcss.append(km.inertia_)

#plt.plot(range(1, 20), wcss)

#Model
X = df.iloc[:,:].values
km = KMeans(n_clusters=4)
y_means = km.fit_predict(X)

plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color='blue')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color='red')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color='green')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1],color='yellow')
plt.show()


