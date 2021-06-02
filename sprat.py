#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, CARTODBPOSITRON
from bokeh.models.tools import WheelZoomTool, BoxSelectTool
from bokeh.models import ColumnDataSource, LinearColorMapper

import streamlit as st

# page configuration
st.set_page_config(
page_title="BoRea Xplore v1.0",
layout="wide",
)

# title
st.title('Sprattus sprattus (1850-2017)')

"""
### Source de la méthode de visualisation des données :
[consultable ici](https://github.com/KikalaT/borea/blob/main/sprat.ipynb)
"""
# ~ csv_file = st.file_uploader('Téléversez votre fichier CSV')

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def preprocessing():

	df_data = pd.read_csv('https://www.jazzreal.org/static/export_data_sprat_data.csv', index_col=0)
	k = 6378137
	df_data = df_data.sample(frac=0.1,replace=False)
	df_data["x"] = df_data['Longitude'] * (k * np.pi / 180.0)
	df_data["y"] = np.log(np.tan((90 + df_data['Latitude']) * np.pi / 360.0)) * k
	return df_data

df_ = preprocessing()

annee = st.selectbox('Année',df_.columns.unique()[2:])

# création des sources de travail
df_years = pd.DataFrame(df_.loc[:,annee])
df_gps = pd.DataFrame(df_.loc[:,['x','y']]) 
df_show = pd.concat([df_gps,df_years], axis=1)
df_mean = pd.DataFrame({'mean':df_.loc[:,'1850':'2017'].mean(axis=1)})
df_annee = pd.DataFrame({'value':df_.loc[:,annee]})

# chargement du fond de carte
tile_provider = get_provider(CARTODBPOSITRON)

# Création de la figure

p = figure(x_range=(-16000000, 16000000), y_range=(-1600000, 16000000),
		   x_axis_type="mercator", y_axis_type="mercator",
		   plot_width=1000,
		   plot_height=800,
		   tools = "pan,wheel_zoom,box_select,box_zoom,reset,save",
		   title='Modélisation : Sprattus sprattus sur la période : '+annee
		   )

p.add_tile(tile_provider)

# source
s1 = ColumnDataSource(data=df_show)

# gradient de couleurs
color_mapper = LinearColorMapper(palette='Turbo256', low=df_show[annee].min(), high=df_show[annee].max())

# points
p.scatter(x='x', y='y', color={'field': annee, 'transform': color_mapper}, alpha=0.7, source=s1)

# paramètres de la figure
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
p.xaxis.major_label_text_color = None
p.yaxis.major_label_text_color = None
p.xaxis.major_tick_line_color = None  
p.xaxis.minor_tick_line_color = None  
p.yaxis.major_tick_line_color = None  
p.yaxis.minor_tick_line_color = None  
p.yaxis.axis_line_color = None
p.xaxis.axis_line_color = None

st.bokeh_chart(p)
