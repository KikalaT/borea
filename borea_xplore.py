# modules
import numpy as np
import pandas as pd
from io import StringIO
from datetime import date, timedelta

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


sns.set()

st.title('BOREA-XPLORE')

"""
## Présentation :
* Le Laboratoire de Biologie des Organismes et des Ecosystèmes Aquatiques - BOREA - a pour objectif l’étude de l’écologie et de la biologie des organismes et des habitats aquatiques dans des écosystèmes naturels et contraints.
* Il s’agit de comprendre, par une approche multidisciplinaire et intégrative, l’origine, le rôle et les mécanismes de l’évolution de la biodiversité aquatique (des molécules aux écosystèmes), les interactions des organismes entre eux et avec leurs milieux de vie et les réponses aux changements globaux, anthropiques et climatiques. 

source : Données d’abondance d’espèces macrobenthiques provenant de la station de prélèvements biologiques de Pierre Noire, en Baie de Morlaix.
"""

# load data (pierre noire)
df = pd.read_csv('pierre-noire-complet.csv',index_col=0)
df = df.T

# load data (somlit)
df_somlit = pd.read_csv('data_somlit_roscoff.csv',sep=';', skiprows=2)
df_somlit_headers = df_somlit.iloc[0,:]
df_somlit = df_somlit.drop(axis=0, labels=0)
df_somlit['DATE'] = pd.to_datetime(df_somlit['DATE'])
df_somlit = df_somlit.set_index('DATE')
num_vars = ['T', 'S', 'O', 'NH4', 'NO3', 'NO2', 'PO4', 'SIOH4', 'COP', 'NOP', 'MES', 'CHLA']
df_somlit = df_somlit.loc[:,num_vars]


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
st.sidebar.markdown('__Séries temporelles__')

specie1 = st.sidebar.selectbox(
    'Espèce 1',
     top25.index
     )
specie2 = st.sidebar.selectbox(
    'Espèce 2',
     top25.index
     )
specie_hue = st.sidebar.selectbox(
    'Espèce 3',
     top25.index
     )

st.sidebar.markdown('__ACP, T-SNE, LDA__') 

specie_target = st.sidebar.selectbox(
    'Espèce cible',
     top25.index
     )
     
st.sidebar.markdown('__K-Means__') 

n_clusters = st.sidebar.selectbox(
    'Nombre de clusters',
     range(2,10)
     )

st.sidebar.markdown('__Corrélation Espèce / Paramètre (Somlit)__')

specie_corr = st.sidebar.selectbox(
    'Espèce',
     top25.index
     )
     
param_corr = st.sidebar.selectbox(
    'Donnée physico-chimique',
     df_somlit.columns
     )
     
"""
#### Histogramme des 25 espèces les plus représentées avec occurrences
---
"""
## disable warnings for st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# histogramme du top25
def plot_hist1():
	plt.figure(figsize=(15,6))
	sns.barplot(x=top25.index, y=top25.values)
	plt.xticks(rotation=45)
	st.pyplot()
plot_hist1()

# data cleaning
df = df.reset_index()
df = df.rename(columns={'index':'date'})
df['date'] = pd.to_datetime(df['date'])

"""
#### Matrice de corrélation entre les 25 espèces les plus représentées
---
"""
@st.cache(suppress_st_warning=True)
def plot_figure1():
	plt.figure(figsize=(18,11))
	sns.heatmap(df.iloc[:,0:24].corr(),annot=True,cmap='viridis')
	st.pyplot()
plot_figure1()

"""
#### Représentation des occurrences au cours des années***
---
"""
'Espèces : ',specie1,specie2,specie_hue

def plot_figure2():
	data = ColumnDataSource(df)
	f2 = figure(plot_width=800,plot_height=400, x_axis_type="datetime")
	r1 = f2.line(source=data,x='date', y=specie1,color='blue', legend_label=specie1, name=specie1)
	r2 = f2.line(source=data,x='date', y=specie2, color='green', legend_label=specie2, name=specie2)
	r3 = f2.line(source=data,x='date', y=specie_hue, color='orange', legend_label=specie_hue, name=specie_hue)
	h1 = HoverTool(renderers = [r1,r2,r3],tooltips = [("espèce","$name"),("date","@date{%F}"),("valeur","$y{0}")],formatters={"@date":"datetime"})
	f2.add_tools(h1)
	f2.legend.click_policy='hide'
	st.bokeh_chart(f2)
plot_figure2()

"""
#### Distribution par espèces***
---
"""

def plot_figure3():
	plt.figure(figsize=(20,9))
	for i,var in enumerate([specie1,specie2,specie_hue]):
		plt.subplot(2,4,i+1)
		sns.histplot(data=df, x=var, kde=True)
		plt.title(var)
		plt.xlabel('')
	st.pyplot()
plot_figure3()

"""
#### Boxplot par espèces***
---
"""
def plot_figure4():
	for i,var in enumerate([specie1,specie2,specie_hue]):
		plt.subplot(2,4,i+1)
		sns.boxenplot(data=df, y=var)
		plt.title(var)
		plt.xlabel('')
	
	st.pyplot()
plot_figure4()

"""
#### Relations entre espèces***
--- 
"""
@st.cache(suppress_st_warning=True)
def plot_figure5():
	sns.scatterplot(data=top25_df, x=specie1, y=specie2, hue=specie_hue,palette='viridis')
	st.pyplot()
plot_figure5()

"""
## Analyse par composantes principales
---
* méthode de réduction de dimension qui consiste à transformer des variables corrélées en nouvelles variables décorrélées les unes des autres.
* Il s’agit de résumer l’information contenue dans un ensemble de données en un certain nombre de variables synthétiques, combinaisons linéaires des variables originelles : ce sont les Composantes Principales. 
* Si l'on adopte un point de vue un peu plus mathématique, l'idée est de projeter l'ensemble des données sur l'hyperplan le plus proche des données. Les vecteurs directeurs de cet hyperplan sont les Composantes Principales.
"""
'Cible : ',specie_target

df_off_target = top25_df.drop(columns=specie_target)
target = top25_df[specie_target]

sc = StandardScaler()
Z = sc.fit_transform(df_off_target)

pca = PCA()
Coord = pca.fit_transform(Z)

"""
### Somme cumulative représentant le ratio de la variance expliquée en fonction du nombre de composantes.
---
"""

len_pca = len(pca.explained_variance_ratio_.cumsum())

def plot_figure6():
	sns.lineplot(x=range(len_pca),
					y=pca.explained_variance_ratio_.cumsum(),
					color='green')
	plt.xticks(range(1,len_pca+1),labels=range(1,len_pca+1))
	st.pyplot()
plot_figure6()

"""
### Camembert la part de variance expliquée par chaque axe de l'ACP
---
"""

def plot_figure7():
	pie_data = pca.explained_variance_
	pie_data = np.concatenate((pie_data[:5],pie_data[6:].sum()),axis=None)
	labels=['C1','C2','C3','C4','C5','6+']
	explode=(0.1,0,0,0,0,0)
	plt.pie(pie_data,startangle=90,autopct='%1.1f%%',shadow=True,explode=explode)
	plt.legend(loc='best',labels=labels)
	st.pyplot()
plot_figure7()

"""
### Coefficient de corrélation de chaque variable de l'ACP sur les deux premiers axes
---
"""
'Cible : ',specie_target

@st.cache(suppress_st_warning=True)
def plot_figure8():
	Comp_PCA = pd.DataFrame({'PC1':pca.components_[:,0],'PC2':pca.components_[:,1]})
	sns.heatmap(Comp_PCA, annot=True, cmap='viridis')
	st.pyplot()
plot_figure8()

"""
### Cercle des corrélations
---
"""
@st.cache(suppress_st_warning=True)
def plot_figure9():
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
plot_figure9()
"""
## T-SNE avec K-MEANS associé***
---
* représentation des coordonnées dans un plan (AXE_1 / AXE_2) en fonction de l'espèce 'cible'
"""
'Cible : ',specie_target

@st.cache(suppress_st_warning=True)
def plot_figure10():
	
	#TSNE
	tsne_new=TSNE(n_components=2,random_state=0)
	Coord_TSNE_ACP=tsne_new.fit_transform(Coord)
	tsne_acp_df=pd.DataFrame({'AXE_1':Coord_TSNE_ACP[:,0],
							'AXE_2':Coord_TSNE_ACP[:,1],
							specie_target:target})
	ax = sns.scatterplot(data=tsne_acp_df,x='AXE_1',y='AXE_2',hue=specie_target,palette='viridis');
	#K-Means
	df_kmeans_tsne=pd.DataFrame(tsne_acp_df,columns=['AXE_1','AXE_2'])
	kmeans_tsne=KMeans(n_clusters=n_clusters)
	kmeans_tsne.fit(df_kmeans_tsne)
	y_kmeans=kmeans_tsne.predict(df_kmeans_tsne)
	#datapre-processing
	y_kmeans_s=pd.Series(y_kmeans,name='cluster')
	df_=pd.concat([df_kmeans_tsne,y_kmeans_s],axis=1)
	#figure
	for i in range(len(kmeans_tsne.cluster_centers_)):
		circle = plt.Circle(xy=(kmeans_tsne.cluster_centers_[i,0], kmeans_tsne.cluster_centers_[i,1]), radius=0.5, color='orange', fill=True, alpha=0.5)
		ax.add_patch(circle)
	st.pyplot(plt)
plot_figure10()

"""
## LDA (Analyse par Discriminant Linéaire)***
---
* représentation des coordonnées dans un plan (AXE_1 / AXE_2) en fonction de l'espèce 'cible'
"""

'Cible : ',specie_target

@st.cache(suppress_st_warning=True)
def plot_figure11():
	lda = LDA()
	X_LDA = lda.fit_transform(df_off_target,target)

	lda_df = pd.DataFrame({'AXE_1':X_LDA[:,0],
							   'AXE_2':X_LDA[:,1],
							   specie_target:target})
							   
	scatter = sns.scatterplot(data=lda_df, x='AXE_1',y='AXE_2',hue=specie_target)
	plt.title('Analyse par Discriminant Linéaire')
	plt.ylabel('AXE_1')
	plt.xlabel('AXE_2')
	scatter.set_xlim([-25,25])

	st.pyplot()
plot_figure11()

"""
## Corrélation entre Espèces et Données physico-chimiques
---
* issus des données physiques, chimiques de la colonne d’eau (SOMLIT)
"""


def plot_figure12():
	# load data
	df1 = top25_df
	df2 = df_somlit
	# data cleaning
	df1 = df1[~df1.index.duplicated()]
	df2 = df2[~df2.index.duplicated()]
	df1[specie_corr] = df1[specie_corr].astype('int64')
	df2[param_corr] = df2[param_corr].astype('float64')
	df2 = df2[~((df2[param_corr] == 999999) | (df2[param_corr] == 999996))]
	df1.index = pd.to_datetime(df1.index)
	df2.index = pd.to_datetime(df2.index)
	df_merged = df1.merge(df2, how='outer', left_index=True, right_index=True)
	df_merged['DATE_M_Y'] = df_merged.index.to_period('M')
	df_ = pd.DataFrame({specie_corr:df_merged.groupby('DATE_M_Y')[specie_corr].sum(),
						param_corr:df_merged.groupby('DATE_M_Y')[param_corr].mean()
						})
	# bokeh figure
	p = figure(plot_width=800,plot_height=600)
	r = p.circle(source=df_,
         x=specie_corr,
         y=param_corr, 
         size=20, 
         color="navy", 
         alpha=0.5)
	p.xaxis.axis_label = specie_corr
	p.yaxis.axis_label = param_corr
	h = HoverTool(renderers = [r],tooltips = [("date","@DATE_M_Y{%F}")],formatters={"@DATE_M_Y":"datetime"})
	p.add_tools(h)
	st.bokeh_chart(p)
plot_figure12()
     

