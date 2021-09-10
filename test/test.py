## 4.6 Test Automatizzato del modello ML ##

# Librerie utili per l'analisi dei dati
from numpy.random.mtrand import seed
import pandas as pd
import numpy as np

# Algoritmi di Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression

# Altre librerie
import pickle
import json

# Configurazione del dataset e suddivisione dei dati in training e test set
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# main
if __name__== "__main__":

    # Lettura dei dati e suddivisione dei dati con ratio 0.2
    df = pd.read_csv('data/data.csv')
    train, test = data_split(df, 0.2)
    X_train = train[['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
       'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
       'Attended Large Gathering', 'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market']].to_numpy()

    # X_test = test[['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
    #    'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
    #    'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
    #    'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
    #    'Attended Large Gathering', 'Visited Public Exposed Places',
    #    'Family working in Public Exposed Places', 'Wearing Masks',
    #    'Sanitization from Market']].to_numpy()

    Y_train = train[['COVID-19']].to_numpy().reshape(4348,)
    # Y_test = test[['COVID-19']].to_numpy().reshape(1086,)

    # Generate some data for validation
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    X_test, y = make_regression(1000, n_features=11)
    
    # Test on the model
    y_hat = clf.predict(X_test)
    

