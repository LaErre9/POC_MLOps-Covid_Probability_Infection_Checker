# Librerie utili per l'analisi dei dati
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Configurazione dello stile dei grafici
sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'

# --- Preparazione dei dati ---
# Caricamento del dataset
covid = pd.read_csv('data.csv')

# Questo metodo stampa le informazioni su un DataFrame, inclusi l'indice dtype e le colonne, i valori non null e l'utilizzo della memoria.
#covid.info()

# Verifica dei dati mancanti
missing_values=covid.isnull().sum() # valori mancanti
percent_missing = covid.isnull().sum()/covid.shape[0]*100 # valori mancanti %
value = {
    'missing_values ':missing_values,
    'percent_missing %':percent_missing  
}
frame=pd.DataFrame(value)
frame.to_csv('Missingvalue.csv') # salvataggio su un file.csv per renderlo leggibile


# Genera statistiche descrittive
#covid.describe().to_csv("dataset_statics.csv") # salvataggio su un file.csv per renderlo leggibile

# --- Visualizzazione dei dati ---

# COVID-19
# sns_plot = sns.countplot(x='COVID-19', data=covid)
# figure = sns_plot.get_figure()
# figure.savefig('COVID-19.png', dpi = 400)

# # Breathing Problem 
# sns_breathing = sns.countplot(x='Breathing Problem',hue='COVID-19',data=covid)
# figure1 = sns_breathing.get_figure()
# figure1.savefig('BreathingProblem.png', dpi = 400)

# Fever 
# sns_fever = sns.countplot(x='Fever', hue='COVID-19', data=covid)
# figure2 = sns_fever.get_figure()
# figure2.savefig('Fever.png', dpi = 400)

# Dry Cough
# sns_dry = sns.countplot(x='Dry Cough',hue='COVID-19',data=covid)
# figure3 = sns_dry.get_figure()
# figure3.savefig('dry.png', dpi = 400)

# Sore Throat
# sns_sore = sns.countplot(x='Sore throat',hue='COVID-19',data=covid)
# figure4 = sns_sore.get_figure()
# figure4.savefig('sore.png', dpi = 400)



