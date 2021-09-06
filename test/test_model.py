from _pytest.python_api import approx
from numpy.random.mtrand import seed
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

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
    df = pd.read_csv('data/data.csv')
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
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    
    #Score/Accuracy
    y_pred = clf.predict(X_test)
    
    #Calcolo della prediction
    inputFeatures = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    accuracy_logreg = clf.score(X_test, Y_test)
    
    # Write results to file
    with open("test/test_score_and_prediction.json", 'w') as outfile:
        json.dump({ "Accuracy": accuracy_logreg, "Prediction":infProb}, outfile)

    # open a file, where yu want to store the data
    file = open('model.pkl','wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()