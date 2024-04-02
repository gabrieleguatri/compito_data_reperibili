#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

try:
    # Importa i dati
    dati = pd.read_csv("titanic.csv")
    # Salva i dati
    dati.to_csv("dati_importati.csv", index=False)
    
    # Modifica i dati (se necessario)

    # Grafico di correlazione
    plt.figure(figsize=(10, 8))
    sns.heatmap(dati.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlazione tra variabili")
    plt.show()

    # Grafico distribuzione delle variabili numeriche
    plt.figure(figsize=(8, 6))
    sns.histplot(dati['Age'], bins=30, kde=True, color='blue')
    plt.title("Distribuzione delle età")
    plt.xlabel("Età")
    plt.ylabel("Frequenza")
    plt.show()

    # Gestione dei missing values
    plt.figure(figsize=(10, 6))
    dati.isnull().sum().plot(kind='bar')
    plt.title("Valori mancanti per variabile")
    plt.xlabel("Variabile")
    plt.ylabel("Numero di valori mancanti")
    plt.xticks(rotation=45)
    plt.show()

    # Grafico degli outliers
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=dati['Age'])
    plt.title("Boxplot per l'età")
    plt.ylabel("Età")
    plt.show()

    # Scaling ed encoding
    scalatore = StandardScaler()
    codificatore = LabelEncoder()

    dati[['Age', 'Fare']] = scalatore.fit_transform(dati[['Age', 'Fare']])
    dati['Sex'] = codificatore.fit_transform(dati['Sex'])

    # Splitting del dataset
    X_train, X_test, y_train, y_test = train_test_split(dati.drop('Survived', axis=1), dati['Survived'], test_size=0.2, random_state=42)


# In[ ]:




