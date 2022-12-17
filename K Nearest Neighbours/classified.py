import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('D:\Datasets/Classified Data', index_col=0)

#How data is Distributes
sns.pairplot(df, hue = 'TARGET CLASS')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df.drop('TARGET CLASS', axis='columns'))
X = pd.DataFrame(X, columns=df.columns[:-1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df['TARGET CLASS'], train_size=0.8)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=14)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))


# Choose optimum K value
error_rate = []
for i in range(1, 50):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    pred_i = model.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

print('K value : ',error_rate.index(min(error_rate)))


plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

