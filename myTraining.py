from numpy.random.mtrand import seed
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score

#librerie per metriche
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score

# librerie per monitoraggio
from yellowbrick.classifier import DiscriminationThreshold

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
    
    clf = LogisticRegression(tol=1100)
    clf.fit(X_train, Y_train)

    #Score/Accuracy
    y_pred = clf.predict(X_test)

    #Calcolo della prediction
    inputFeatures = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    print(infProb)

    # matrice di confusione
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    # plot_confusion_matrix(clf, X_test, Y_test)
    # plt.show()
    
    accuracy_logreg = clf.score(X_test, Y_test)
    precision = precision_score(Y_test, y_pred)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": accuracy_logreg, "specificity": specificity, "sensitivity":sensitivity, "precision": precision}, outfile)

    # Report test set score
    train_score = clf.score(X_train, Y_train) * 100
    test_score = clf.score(X_test, Y_test) * 100
    print(train_score)
    print(test_score)

    # Scrittura dei scores al file
    with open("score.txt", 'w') as outfile:
            outfile.write("Training variance explained: %2.1f%%\n" % train_score)
            outfile.write("Test variance explained: %2.1f%%\n" % test_score)

    # Visualizzatore
    visualizer_report = DiscriminationThreshold(clf)
    visualizer_report.fit(X_train, Y_train)  
    visualizer_report.score(X_test, Y_test)  
    visualizer_report.show("report.png")     
  
   # ----------------------------------------------------------------------------

    # open a file, where yu want to store the data
    file = open('model.pkl','wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()

