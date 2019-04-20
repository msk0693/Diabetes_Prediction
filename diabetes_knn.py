import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#-Load Data-
data = pd.read_csv('data/diabetes_data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 8].values

#-Isolate Train/Test Set
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.25, random_state =42)

#-Normalization-
#Features should be normalized so that each feature contributes approximately proportionately
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-Param Evaluation-
knnclf = KNeighborsClassifier()
parameters = {'n_neighbors': range(1, 20)}
gridsearch = GridSearchCV(knnclf, parameters, cv=100, scoring='roc_auc')
gridsearch.fit(X,y)
print(gridsearch.best_params_)
print(gridsearch.best_score_)
#-Output-
#BestParam: k=18
#BestScore: 0.8055338541666667

#-Fit algorithm to training data-
knnClassifier = KNeighborsClassifier(n_neighbors=18)
knnClassifier.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knnClassifier.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knnClassifier.score(X_test, y_test)))
#-Output-
#Accuracy of K-NN classifier on training set: 0.79
#Accuracy of K-NN classifier on test set: 0.71

#-Predict test set results-
y_prediction = knnClassifier.predict(X_test)

#-Confusion Matrix-
cm = confusion_matrix(y_test, y_prediction)

print('TP - True Negative {}'.format(cm[0,0]))
print('FP - False Positive {}'.format(cm[0,1]))
print('FN - False Negative {}'
	.format(cm[1,0]))
print('TP - True Positive {}'.format(cm[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))))

print(round(roc_auc_score(y_test,y_prediction),5))
#-Output-
# TP - True Negative 108
# FP - False Positive 15
# FN - False Negative 40
# TP - True Positive 29
# Accuracy Rate: 0.7135416666666666
# Misclassification Rate: 0.2864583333333333
# 0.64917
