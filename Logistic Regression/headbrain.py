import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('headbrain.csv')

plt.scatter(data['Head Size(cm^3)'], data['Brain Weight(grams)'])

x_train, x_test, y_train, y_test = train_test_split(data[['Gender','Age Range','Head Size(cm^3)']], data['Brain Weight(grams)'], test_size=0.1)

model = LogisticRegression()
model.fit(x_train,y_train)

print(model.intercept_)

print(model.coef_)

print(model.predict([[1,1,4261]]))

print(model.score(x_test, y_test))

