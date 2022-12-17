import pandas as pd

data = pd.read_csv("titanic.csv")
df = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare', 'SibSp', 'Parch'], axis='columns')
df['Age'] = df['Age'].fillna(df['Age'].mean())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['Embarked_n'] = encoder.fit_transform(df['Embarked'])


df['Sex_n'] = encoder.fit_transform(df['Sex'])

df = df.drop(['Sex','Embarked'], axis='columns')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived'], axis='columns'), df['Survived'],
                                                    train_size=0.8)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
