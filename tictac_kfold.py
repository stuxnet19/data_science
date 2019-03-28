import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
from sklearn.model_selection import KFold


agaricus_data = pd.read_csv('DataSets/tic-tac-toe.data',
        sep=',',header = None)

X = agaricus_data.values[:,0:8]
Y = agaricus_data.values[:,9]
X_df = pd.DataFrame(data=X)
Y_df = pd.DataFrame(data=Y)
X_dm = pd.get_dummies(X_df)
X_mat = X_dm.as_matrix()
kfold = KFold(10,True,10)
ac = 0.0
ac_score = 0.0

for train,test in kfold.split(X_dm):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=5, min_samples_leaf=5)
    X_train, X_test, y_train, y_test = X_mat[train], X_mat[test], Y[train], Y[test]
    clf_entropy.fit(X_train, y_train)
    y_pred_en = clf_entropy.predict(X_test)
    ac_score=accuracy_score(y_test,y_pred_en)*100
    ac=ac+ac_score
    print ("Accuracy is ", ac_score)

ac_avg=ac/10
print ("Average Accuracy is ", ac_avg)
