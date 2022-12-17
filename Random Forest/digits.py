import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
digits = load_digits()

data = pd.DataFrame(digits.data)

data['target'] = digits.target

X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis='columns'), data['target'], test_size=0.8)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))