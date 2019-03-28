import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

agaricus_data = pd.read_csv('DataSets/tic-tac-toe.data',sep=',',header = None)
x = agaricus_data.values[:,0:8]
y = agaricus_data.values[:,9]
x = pd.DataFrame(data=x)
y = pd.DataFrame(data=y)


x_dum = pd.get_dummies(x)
X_train,X_test,Y_train,Y_test = train_test_split(x_dum,y,test_size=0.3)


print("max_depth = 3 \n")
clf_entropy = DecisionTreeClassifier(criterion= "entropy",
        max_depth=3,min_samples_leaf=5)

clf_entropy.fit(X_train,Y_train)
y_pred_en = clf_entropy.predict(X_test)
print("Accuracy is ",accuracy_score(Y_test,y_pred_en)*100)



print("max_depth = 6 \n")
clf_entropy = DecisionTreeClassifier(criterion= "entropy",
        max_depth=6,min_samples_leaf=5)

clf_entropy.fit(X_train,Y_train)
y_pred_en = clf_entropy.predict(X_test)
print("Accuracy is ",accuracy_score(Y_test,y_pred_en)*100)
