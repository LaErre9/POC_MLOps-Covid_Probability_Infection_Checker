import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from myTraining import X_train, Y_train, X_test, Y_test

   
   
# Accuracy RandomForestRegressor
clf1 = RandomForestRegressor(n_estimators=1000)
clf1.fit(X_train, Y_train)
acc_randomforest = clf1.score(X_test, Y_test)*100

# Accuracy GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
GBR.fit(X_train, Y_train)
acc_gbk=GBR.score(X_test, Y_test)*100

# Accuracy KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
#Score/Accuracy
acc_knn=knn.score(X_test, Y_test)*100

# Accuracy DecisionTreeClassifier
tree = tree.DecisionTreeClassifier()
tree.fit(X_train,Y_train)
y_pred1 = tree.predict(X_test)
acc_decisiontree=tree.score(X_test, Y_test)*100

# Accuracy Naive_bayes
model = GaussianNB()
model.fit(X_train,Y_train)
acc_gaussian= model.score(X_test, Y_test)*100

    
# Classifica del miglior modello in funzione dello Score/Accuracy
models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',   
              'Decision Tree', 'Gradient Boosting Classifier'],
    'Score': [acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_decisiontree,
              acc_gbk]})
print(models.sort_values(by='Score', ascending=False))