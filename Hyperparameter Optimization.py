import pandas as pd

df = pd.read_csv("D:/Datasets/Churn_Modelling.csv")
print(df.head())
import seaborn as sns
import matplotlib.pyplot as plt

#corelation
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
#plt.show()

X = df.drop(['RowNumber', 'CustomerId','Exited', 'Surname'], axis='columns')
y = df['Exited']

geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

X.drop(['Geography', 'Gender'], axis='columns', inplace=True)
X=pd.concat([X, geography, gender], axis='columns')

params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]

}

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost

classifier = xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=4,verbose=3)
random_search.fit(X,y)

print(random_search.best_estimator_)
print(random_search.best_params_)

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=4)

print(score.mean())