import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()
#print(digits.__dir__())

data = pd.DataFrame(digits.data, columns=digits.feature_names)
#print(data.head())

X = data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.decomposition import PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)
#print(X_pca.shape)

X_train, X_test, y_train, y_test = train_test_split(X_pca, digits.target, train_size=0.8)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))