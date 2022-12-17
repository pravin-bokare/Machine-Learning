import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek

df = pd.read_csv('D:/Datasets/creditcard.csv')

#print(df.info())

columns = df.columns.tolist()

columns = [c for c in columns if c not in ['Class']]

target = 'Class'

state = np.random.RandomState()
X = df[columns]
Y = df[target]

X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
print(X.shape)
print(Y.shape)

#EDA

df.isnull().values.any()

count_classes = pd.value_counts(df['Class'], sort=True)

count_classes.plot(kind='bar', rot=0)
plt.title('Transaction class Distribution')
plt.xticks(range(2))
plt.xlabel('Class')
plt.ylabel('Frequency')

fraud = df[df['Class']==1]
normal = df[df['Class']==0]

print(fraud.shape, normal.shape)

from imblearn.over_sampling import RandomOverSampler

# Implementing Oversampling for Handling Imbalanced
smk = SMOTETomek(random_state=42)
X_res,y_res=smk.fit_resample(X,Y)

print(X_res.shape,y_res.shape)