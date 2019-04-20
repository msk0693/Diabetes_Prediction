import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#-Load Data-
data = pd.read_csv('data/diabetes_data.csv')

# print(data.columns)
# print(data.head())
# print(diabetes.info())
print("Dimension of data: {}".format(data.shape))

#Need to predict "Outcome" feature: 0 - No, 1 - Yes
print(data.groupby('Outcome').size())
#Of 768, 500 labeled as 0 and 268 labled as 1

#-Plot "Outcome"-
sns.countplot(data['Outcome'], label="Count")


#-Test connection between model complexity and accuracy-
X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'Outcome'], data['Outcome'], stratify=data['Outcome'], random_state=66)

training_accuracy = []
test_accuracy = []
#try k from 1 to 10
neighbors_settings = range(1, 21)
for n_neighbors in neighbors_settings:
	#build model
	knn = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn.fit(X_train, y_train)
	#training set accuracy
	training_accuracy.append(knn.score(X_train, y_train))
	#test set accuracy
	test_accuracy.append(knn.score(X_test, y_test))

#-Plot-
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.legend()
plt.savefig('knn_compare_model')
plt.show()

#-Output: Plot suggests that we should choose k=18

