import pandas as pd

data = pd.read_csv("diabetes.csv")

# print(data.head())

X = data.drop(['Outcome'], axis='columns')
y = data['Outcome']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8,
                              oob_score=True, random_state=True)

bag_model.fit(X_train, y_train)
print(bag_model.score(X_test, y_test))
