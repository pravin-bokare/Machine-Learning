import pandas as pd
from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

data['target'] = iris.target
print(data.head(5))

df0 = data[:50]
df1 = data[50:100]
df2 = data[100:]

plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'])
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'])
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'])
#plt.show()

X = data.drop(['target'], axis='columns')
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))