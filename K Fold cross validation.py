import pandas
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv("iris.csv")

encoder = LabelEncoder()
data['target'] = encoder.fit_transform(data['variety'])

X = data.drop(['variety', 'target'], axis='columns')
y = data['target']


from sklearn.model_selection import cross_val_score

print("LinearRegression = ",np.average(cross_val_score(LinearRegression(), X, y)))
print("LogisticRegression = ",np.average(cross_val_score(LogisticRegression(solver='liblinear'), X, y)))
print("DecisionTreeClassifier = ",np.average(cross_val_score(DecisionTreeClassifier(), X, y)))
print("RandomForestClassifier = ",np.average(cross_val_score(RandomForestClassifier(), X, y)))
print("NaiveBayes = ",np.average(cross_val_score(GaussianNB(), X, y)))
print("SupportVectorMachine = ",np.average(cross_val_score(SVC(), X, y)))
print("KMeanClustering = ", np.average(cross_val_score(KMeans(n_clusters=3), X, y)))