import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('housing.csv')

df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=df['population']/100, figsize=(10, 7),
        label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
#plt.show()

encoder = LabelEncoder()
df['ocean_proximity'] = encoder.fit_transform(df['ocean_proximity'])
X = df.drop(['median_house_value'], axis='columns')
Y = df['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))