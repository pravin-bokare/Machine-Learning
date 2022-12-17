import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("titanic.csv")
data.Age = data.Age.fillna(data.Age.mean())

ndata = data.drop(['PassengerId','Embarked','Name', 'Cabin', 'Ticket','SibSp','Parch','Ticket','Sex'], axis='columns')


X = ndata.drop(['Survived'], axis='columns')
y = ndata.Survived

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)

model = GaussianNB()
model.fit(X_train,y_train)

print(model.score(X_test, y_test))