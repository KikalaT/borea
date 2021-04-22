# modules
import numpy as np
import pandas as pd
from io import StringIO
from datetime import date, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

st.title('BOREA-XPLORE')

"""
## Présentation :
* Le Laboratoire de Biologie des Organismes et des Ecosystèmes Aquatiques - BOREA - a pour objectif l’étude de l’écologie et de la biologie des organismes et des habitats aquatiques dans des écosystèmes naturels et contraints.
* Il s’agit de comprendre, par une approche multidisciplinaire et intégrative, l’origine, le rôle et les mécanismes de l’évolution de la biodiversité aquatique (des molécules aux écosystèmes), les interactions des organismes entre eux et avec leurs milieux de vie et les réponses aux changements globaux, anthropiques et climatiques. 

source : Données d’abondance d’espèces macrobenthiques provenant de la station de prélèvements biologiques de Pierre Noire, en Baie de Morlaix.
"""

# load data
df = pd.read_csv('pierre-noire-complet.csv',index_col=0)
df = df.T

 
"""
#### Tableau des 25 espèces les plus représentées avec occurrences
---
"""
top25 = df.sum().sort_values(ascending=False)[:24]
top25_df = df.loc[:,top25.index]

# display
top25_df

# sidebar

st.sidebar.header('Outils')
st.sidebar.markdown('__Visualisation__')

specie1 = st.sidebar.selectbox(
    'Espèce 1',
     top25.index
     )
specie2 = st.sidebar.selectbox(
    'Espèce 2',
     top25.index
     )
specie3 = st.sidebar.selectbox(
    'Espèce 3',
     top25.index
     )
specie_hue = st.sidebar.selectbox(
    'Espèce 4',
     top25.index
     )

st.sidebar.markdown('__Analyse en composantes principales (ACP) + T-SNE__') 

specie_target = st.sidebar.selectbox(
    'Espèce cible',
     top25.index
     )
     
     
"""
#### Histogramme des 25 espèces les plus représentées avec occurrences
---
"""
## disable warnings for st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# histogramme du top25

plt.figure(figsize=(15,6))
sns.barplot(x=top25.index, y=top25.values)
plt.xticks(rotation=45)
st.pyplot()

# data cleaning
df = df.reset_index()
df = df.rename(columns={'index':'date'})
df['date'] = pd.to_datetime(df['date'])

"""
#### Représentation des occurrences au cours des années
---
"""
'Espèces : ',specie1,specie2,specie3

plt.figure(figsize=(15,6))
sns.lineplot(data=df, x='date', y=specie1, label=specie1)
sns.lineplot(data=df, x='date', y=specie2, label=specie2)
sns.lineplot(data=df, x='date', y=specie3, label=specie3)

st.pyplot()

"""
#### Distribution par espèces
---
"""

plt.figure(figsize=(20,9))
for i,var in enumerate([specie1,specie2,specie3]):
    plt.subplot(2,4,i+1)
    sns.histplot(data=df, x=var, kde=True)
    plt.title(var)
    plt.xlabel('')
st.pyplot()

"""
#### Boxplot par espèces
---
"""

plt.figure(figsize=(20,9))
for i,var in enumerate([specie1,specie2,specie3]):
    plt.subplot(2,4,i+1)
    sns.boxenplot(data=df, y=var)
    plt.title(var)
    plt.xlabel('')
    
st.pyplot()

"""
#### Matrice de corrélation entre les 25 espèces les plus représentées
---
"""

plt.figure(figsize=(18,11))
sns.heatmap(df.iloc[:,0:24].corr(),annot=True,cmap='viridis')
st.pyplot()

"""
#### Relations entre espèces
--- 
"""

sns.scatterplot(data=top25_df, x=specie1, y=specie2, size=specie3, hue=specie_hue,palette='viridis')
st.pyplot()

"""
## Analyse par composantes principales
---
* méthode de réduction de dimension qui consiste à transformer des variables corrélées en nouvelles variables décorrélées les unes des autres.
* Il s’agit de résumer l’information contenue dans un ensemble de données en un certain nombre de variables synthétiques, combinaisons linéaires des variables originelles : ce sont les Composantes Principales. 
* Si l'on adopte un point de vue un peu plus mathématique, l'idée est de projeter l'ensemble des données sur l'hyperplan le plus proche des données. Les vecteurs directeurs de cet hyperplan sont les Composantes Principales.
"""
'Cible : ',specie_target


df_ = top25_df.drop(columns=specie_target)
target = top25_df[specie_target]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Z = sc.fit_transform(df_)

from sklearn.decomposition import PCA
pca = PCA()
Coord = pca.fit_transform(Z)

"""
### Somme cumulative représentant le ratio de la variance expliquée en fonction du nombre de composantes.
---
"""
len_pca = len(pca.explained_variance_ratio_.cumsum())
sns.lineplot(x=range(len_pca),
				y=pca.explained_variance_ratio_.cumsum(),
				color='green')
plt.xticks(range(1,len_pca+1),labels=range(1,len_pca+1))
st.pyplot()

"""
### Camembert la part de variance expliquée par chaque axe de l'ACP
---
"""

# data
pie_data = pca.explained_variance_
pie_data = np.concatenate((pie_data[:5],pie_data[6:].sum()),axis=None)
# figure
labels=['C1','C2','C3','C4','C5','6+']
explode=(0.1,0,0,0,0,0)
plt.pie(pie_data,startangle=90,autopct='%1.1f%%',shadow=True,explode=explode)
plt.legend(loc='best',labels=labels)
st.pyplot()

"""
### Coefficient de corrélation de chaque variable de l'ACP sur les deux premiers axes
---
"""

Comp_PCA = pd.DataFrame({'PC1':pca.components_[:,0],'PC2':pca.components_[:,1]})
sns.heatmap(Comp_PCA, annot=True, cmap='viridis')
st.pyplot()

"""
### Cercle des corrélations
---
"""
racine_valeurs_propres=np.sqrt(pca.explained_variance_)
corvar=np.zeros((len_pca,len_pca))
for k in range(len_pca):
	corvar[:,k]=pca.components_[:,k]*racine_valeurs_propres[k]

# Délimitation de la figure
fig,axes=plt.subplots(figsize=(10,10))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
# Affichage des variables
for j in range(len_pca):
	plt.annotate(top25_df.columns[j],(corvar[j,0],corvar[j,1]),color='#091158')
	plt.arrow(0,0,corvar[j,0]*0.6,corvar[j,1]*0.6,alpha=0.5,head_width=0.03,color='b')

# Ajout des axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

# Cercle et légendes
cercle=plt.Circle((0,0),1,color='#16E4CA',fill=False)
axes.add_artist(cercle)
plt.xlabel('AXE1')
plt.ylabel('AXE2')

# display
st.pyplot()

"""
### ACP + algorithme T-SNE (t-distributed stochastic neighbor embedding)
---
* représentation des coordonnées dans un plan (AXE_1 / AXE_2) en fonction de l'espèce 'cible'
"""
'Cible : ',specie_target

from sklearn.manifold import TSNE
tsne_new = TSNE(n_components=2, random_state=0)
Coord_TSNE_ACP = tsne_new.fit_transform(Coord)
tsne_acp_df = pd.DataFrame({'AXE_1':Coord_TSNE_ACP[:,0],
                           'AXE_2':Coord_TSNE_ACP[:,1],
                           specie_target:target})

sns.scatterplot(data=tsne_acp_df,x='AXE_1',y='AXE_2',hue=specie_target,palette='viridis');

st.pyplot()
