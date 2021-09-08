## 4.4.5 Test di predizione del modello ML ##

# Librerie utili per l'analisi dei dati
from numpy.random.mtrand import seed
import pandas as pd
import numpy as np

# Algoritmi di Machine Learning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression

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

    X_test = test[['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
       'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
       'Attended Large Gathering', 'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market']].to_numpy()

    Y_train = train[['COVID-19']].to_numpy().reshape(4348,)
    Y_test = test[['COVID-19']].to_numpy().reshape(1086,)

    # Predizione e utilizzo dell'algoritmo di Machine Learning
    clf = LogisticRegression()  # Puoi modificare in LogisticRegressioneCV e verificare le differenze con LogisticRegression
    clf.fit(X_train, Y_train)
    
    # Calcolo della prediction
    # I valori passati indicano un paziente con: problema di respirazione (breathing problem), febbre (fever), tosse secca (dry cough), 
    # presenta una malattia cardiaca (heart disease), ipertensione (hyper tension) e mostra fatica (fatigue). 
    inputFeatures = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    accuracy_logreg = clf.score(X_test, Y_test)
    
    # Scrittura su file .json
    with open("test/test_score_and_prediction.json", 'w') as outfile:
        json.dump({ "Accuracy": accuracy_logreg, "Prediction":infProb}, outfile)

    # open a file, where yu want to store the data
    file = open('model.pkl','wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()

