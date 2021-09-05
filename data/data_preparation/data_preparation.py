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

# Questo metodo stampa le informazioni su un DataFrame, inclusi l'indice dtype e le colonne, i valori non null e l'utilizzo della memoria.
# Scrittura delle info su file.txt
with open('data/data_preparation/info.txt', 'w') as f:
    covid.info(buf=f)


# Verifica dei dati mancanti
missing_values=covid.isnull().sum() # valori mancanti
percent_missing = covid.isnull().sum()/covid.shape[0]*100 # valori mancanti %
value = {
    'missing_values ':missing_values,
    'percent_missing %':percent_missing  
}
frame=pd.DataFrame(value)
frame.to_csv('data/data_preparation/missing_value.csv') # salvataggio su un file.csv per renderlo leggibile


# Genera statistiche descrittive
covid.describe().to_csv("data/data_preparation/dataset_statics.csv") # salvataggio su un file.csv per renderlo leggibile

# --- Visualizzazione dei dati ---
# COVID-19
# sns_plot = sns.countplot(x='COVID-19', data=covid)
# figure = sns_plot.get_figure()
# figure.savefig('data/data_preparation/data_viz/COVID-19.png', dpi = 400)

# Breathing Problem 
# sns_breathing = sns.countplot(x='Breathing Problem',hue='COVID-19',data=covid)
# figure1 = sns_breathing.get_figure()
# figure1.savefig('data/data_preparation/data_viz/BreathingProblem.png', dpi = 400)

# Fever 
# sns_fever = sns.countplot(x='Fever', hue='COVID-19', data=covid)
# figure2 = sns_fever.get_figure()
# figure2.savefig('data/data_preparation/data_viz/Fever.png', dpi = 400)

# Dry Cough
# sns_dry = sns.countplot(x='Dry Cough',hue='COVID-19',data=covid)
# figure3 = sns_dry.get_figure()
# figure3.savefig('data/data_preparation/data_viz/dry.png', dpi = 400)

# Sore Throat
# sns_sore = sns.countplot(x='Sore throat',hue='COVID-19',data=covid)
# figure4 = sns_sore.get_figure()
# figure4.savefig('data/data_preparation/data_viz/sore.png', dpi = 400)




