import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv')
data.fillna(value=data['Age'].mean())
inputs = data.drop(['Name', 'Ticket', 'Survived', 'Cabin', 'Embarked','PassengerId','SibSp', 'Parch'], axis='columns')

le_sex = LabelEncoder()
inputs['S'] = le_sex.fit_transform(inputs['Sex'])
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

X = inputs.drop(['Sex'], axis='columns')
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print(model.score(X_test,y_test))
