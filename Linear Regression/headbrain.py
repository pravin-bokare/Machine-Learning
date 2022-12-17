import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv('headbrain.csv')

plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.scatter(data['Head Size(cm^3)'], data['Brain Weight(grams)'])
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(data[['Gender','Age Range','Head Size(cm^3)']], data['Brain Weight(grams)'], test_size=0.1)

model = LinearRegression()
model.fit(x_train, y_train)

print(model.intercept_)

print(model.coef_)

print(model.predict([[1,1,4261]]))

print(model.score(x_test, y_test))

