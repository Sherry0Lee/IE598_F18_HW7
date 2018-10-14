#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:18:03 2018

@author: sherry
"""
import sklearn
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

df = pd.read_csv('/Users/sherry/Downloads/wine.csv')
#df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids',  'Nonflavanoid phenols',  'Proanthocyanins', 'Color intensity', 'Hue',  'OD280/OD315 of diluted wines', 'Proline']
#print('Class labels', np.unique(df['Class label']))
df.head()
X, y = df.iloc[:, 1:].values,df.iloc[:, 13].values
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.1,stratify=y)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators=500)
for i in [1,2,5,10,25,50,100,500]:
    forest.n_estimators=i

#    y_pred=forest.predict(X_test)
    score=cross_val_score(estimator=forest,X=X_train,y=y_train,cv=10,scoring='accuracy')
#    score=metrics.accuracy_score(y_test, y_pred)
    print(np.mean(score))
    
    
forest.fit(X_train, y_train)       
feat_labels = df.columns[:-1]

from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators':[1,2,5,10,25,50,100,500]}
gs = GridSearchCV(estimator=forest,param_grid=param_grid,scoring='accuracy',cv=10)
gs.fit(X_train,y_train)
gs.best_estimator_.fit(X_train, y_train)
print(gs.best_params_)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
