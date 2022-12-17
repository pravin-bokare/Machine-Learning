import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')
X = data.drop(['Class'], axis='columns')
y = data.Class

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = LogisticRegression()

model.fit(x_train, y_train)

print(model.score(x_test, y_test)*100)

