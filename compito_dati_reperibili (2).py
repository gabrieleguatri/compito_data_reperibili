#!/usr/bin/env python
# coding: utf-8

# # POKEMON

# # splitting dataset

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specifica il percorso del tuo file CSV
percorso_file_csv ="C:\\Users\\utente\\Pokemons.csv"

# Leggi il file CSV in un DataFrame
df = pd.read_csv(percorso_file_csv)

# Mostra le prime righe del DataFrame (opzionale)
print(df.head())
df.shape


# #  grafico missing valeus

# In[25]:


import matplotlib.pyplot as plt

# Dati dei valori mancanti
colonne = ['HP', 'Attack', 'Defense', 'Speed', 'Total']
valori_mancanti = [0, 7, 10, 0, 0]

# Visualizza un grafico a barre dei valori mancanti
plt.figure(figsize=(10, 6))
plt.bar(colonne, valori_mancanti, color='blue', edgecolor='black')
plt.title('Valori Mancanti nel Dataset')
plt.xlabel('')
plt.ylabel('Numero di Valori Mancanti')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# # grafico outliers

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importa il dataset
pokemon_data = pd.read_csv('C:\\Users\\utente\\Pokemons.csv')

# Calcola la media e la deviazione standard della colonna 'HP'
mean_value = pokemon_data['hp'].mean()
std_dev = pokemon_data['hp'].std()

# Trova gli outliers
outliers = pokemon_data[(pokemon_data['hp'] > mean_value + 3 * std_dev) | (pokemon_data['hp'] < mean_value - 3 * std_dev)]

# Crea un grafico a dispersione
plt.scatter(pokemon_data.index, pokemon_data['hp'], label='hp')

# Evidenzia gli outliers nel grafico con un colore diverso
plt.scatter(outliers.index, outliers['hp'], color='red', label='Outliers')

# Aggiungi linee per la media e la deviazione standard
plt.axhline(y=mean_value, color='green', linestyle='--', label='Media')
plt.axhline(y=mean_value + 3 * std_dev, color='orange', linestyle='--', label='±3 Deviazioni Standard')
plt.axhline(y=mean_value - 3 * std_dev, color='orange', linestyle='--')

# Aggiungi etichette e legenda
plt.xlabel('Indice')
plt.ylabel('hp')
plt.title('Grafico con Outliers Evidenziati')
plt.legend()

# Mostra il grafico
plt.show()


# # scalign ed encoding

# In[4]:


print(pokemon_data.columns)


# In[6]:


print(pokemon_data.head())


# In[23]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Importa il dataset
pokemon_data = pd.read_csv('C:\\Users\\utente\\Pokemons.csv')

# Encoding delle variabili categoriche
label_encoder = LabelEncoder()
pokemon_data['Legendary'] = label_encoder.fit_transform(pokemon_data['rank'])

# Scaling dei dati
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pokemon_data[['hp']])

# Grafico delle distribuzioni dopo l'encoding e lo scaling
plt.figure(figsize=(12, 6))
sns.histplot(scaled_data, color='blue', bins=30)
plt.title('Distribuzione dei Punti Salute (hp) dopo encoding e scaling')
plt.xlabel('hp (Scaled)')
plt.ylabel('Frequenza')
plt.show()


# In[ ]:





# In[ ]:




