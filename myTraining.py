import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

#librerie per metriche
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score

import pickle
import json

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__== "__main__":

    # Read The Data
    df = pd.read_csv('data.csv')
    train, test = data_split(df, 0.2)
    X_train = train[['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
       'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
       'Attended Large Gathering', 'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market']].to_numpy()

    X_test = test[['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
       'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
       'Attended Large Gathering', 'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market']].to_numpy()

    Y_train = train[['COVID-19']].to_numpy().reshape(4348,)
    Y_test = test[['COVID-19']].to_numpy().reshape(1086,)

   # ------------ LogisticRegression ---------------------------------------------
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)

    #Score/Accuracy
    y_pred = clf.predict(X_test)

    #Calcolo della prediction
    inputFeatures = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    print(infProb)

    # matrice di confusione
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    # matrice di confusione a video
    plot_confusion_matrix(clf, X_test, Y_test)
    plt.show()
    acc_logreg = clf.score(X_test, Y_test)
    precision = precision_score(Y_test, y_pred)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    

    with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc_logreg, "specificity": specificity, "sensitivity":sensitivity, "precision": precision}, outfile)

   # ----------------------------------------------------------------------------
   
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
    #print(models.sort_values(by='Score', ascending=False))


    # open a file, where yu want to store the data
    file = open('model.pkl','wb')

    # with open('model.pkl', 'rb') as f:
    # data = pickle.load(f)
   
    # dump information to that file
    pickle.dump(clf, file)
    file.close()

