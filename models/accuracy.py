# Librerie utili per l'analisi dei dati
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random.mtrand import seed

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree

# Configurazione dello stile dei grafici
sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'

# --- Preparazione dei dati ---
# Caricamento del dataset

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__== "__main__":

    # Read The Data
    covid = pd.read_csv('data/data.csv')
    train, test = data_split(covid, 0.2)
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
    
# Analisi delle accuratezze di ogni algoritmo di apprendimento
# Accuracy LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, Y_train)
accuracy_logreg = clf.score(X_test, Y_test)*100

# Accuracy RandomForestRegressor
clf1 = RandomForestRegressor(max_depth = 2, n_estimators=1000)
clf1.fit(X_train, Y_train)
accuracy_randomforest = clf1.score(X_test, Y_test)*100

# Accuracy GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
GBR.fit(X_train, Y_train)
accuracy_gbk=GBR.score(X_test, Y_test)*100

# Accuracy KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
#Score/Accuracy
accuracy_knn=knn.score(X_test, Y_test)*100

# Accuracy DecisionTreeClassifier
tree = tree.DecisionTreeClassifier()
tree.fit(X_train,Y_train)
y_pred1 = tree.predict(X_test)
accuracy_decisiontree=tree.score(X_test, Y_test)*100

# Accuracy Naive_bayes
model = GaussianNB()
model.fit(X_train,Y_train)
accuracy_gaussian= model.score(X_test, Y_test)*100
    
# Classifica del miglior modello in funzione dello Score/Accuracy
models = pd.DataFrame({
   'Model': ['KNN', 'Logistic Regression', 
             'Random Forest', 'Naive Bayes',   
             'Decision Tree', 'Gradient Boosting Classifier'],
   'Score': [accuracy_knn, accuracy_logreg, 
             accuracy_randomforest, accuracy_gaussian, accuracy_decisiontree,
             accuracy_gbk]})
classifica = models.sort_values(by='Score', ascending=False)
classifica.to_csv("models/classifica_accuracy_model")