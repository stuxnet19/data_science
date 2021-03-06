import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


balance_data = pd.read_csv('DataSets/agaricus-lepiota.data', sep= ',', header= None)
balance_data.shape
X = balance_data.values[:, 1:]
Y = balance_data.values[:,0]
x_dum = pd.get_dummies(pd.DataFrame(X))

print("test ----> 30%")
print("train ----> 70%")

X_train, X_test, y_train, y_test = train_test_split( x_dum,Y, test_size = 0.3)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=3, min_samples_leaf=5)
#X_train = pd.get_dummies(pd.DataFrame(data=X_train))
#X_test = pd.get_dummies(pd.DataFrame(data=X_test))
clf_entropy.fit(X_train, y_train)
y_pred_en = clf_entropy.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
print("train ------> 50%")
print("test -------> 50%")
X_train, X_test, y_train, y_test = train_test_split( x_dum, Y, test_size = 0.5)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=3, min_samples_leaf=5)
#X_train = pd.get_dummies(pd.DataFrame(data=X_train))
#X_test = pd.get_dummies(pd.DataFrame(data=X_test))
clf_entropy.fit(X_train, y_train)
y_pred_en = clf_entropy.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)


