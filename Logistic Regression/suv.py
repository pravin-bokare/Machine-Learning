import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.read_csv('suv_data.csv')

x = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

#model.predict(X_test, y_test)

print(accuracy_score(y_test, predictions))
