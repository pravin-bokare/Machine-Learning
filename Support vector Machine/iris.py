import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)

data['target'] = iris['target']
#print(data.head())

X = data.drop(['target'], axis='columns')
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = SVC()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))