#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, CARTODBPOSITRON
from bokeh.models.tools import WheelZoomTool
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.io import show, output_notebook

import streamlit as st

st.title('Sprattus sprattus (1850-2017)')

"""
### Source de la méthode d'analyse des données :
[consultable ici](https://github.com/KikalaT/borea/blob/main/sprat.ipynb)
"""
# ~ csv_file = st.file_uploader('Téléversez votre fichier CSV')

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def preprocessing():
	
	df_data = pd.read_csv('https://www.jazzreal.org/static/export_data_sprat_data.csv', index_col=0)
	
	return df_data
	
df_ = preprocessing()

"""
### Aperçu des données modélisées :
"""

k = 6378137
df_["x"] = df_['Longitude'] * (k * np.pi / 180.0)
df_["y"] = np.log(np.tan((90 + df_['Latitude']) * np.pi / 360.0)) * k

st.write(df_.head())

"""
### Géolocalisation des données :
"""

annee = st.selectbox('Année',df_.columns.unique()[2:])

# création de la source
df_years = pd.DataFrame(df_.loc[:,annee])
df_gps = pd.DataFrame(df_.loc[:,['x','y']]) 
df_show = pd.concat([df_gps,df_years], axis=1)

# chargement du fond de carte
tile_provider = get_provider(CARTODBPOSITRON)

# Création de la figure

p = figure(x_range=(-16000000, 16000000), y_range=(-1600000, 16000000),
		   x_axis_type="mercator", y_axis_type="mercator",
		   plot_width=800,
		   plot_height=600,
		   tools = "pan,wheel_zoom,box_select,box_zoom,reset,save",
		   title='Modélisation : Sprattus sprattus sur la période : '+annee
		   )

p.add_tile(tile_provider)

# source
geo_source = ColumnDataSource(data=df_show)

# gradient de couleurs
color_mapper = LinearColorMapper(palette='Turbo256', low=df_show[annee].min(), high=df_show[annee].max())

# points
p.scatter(x='x', y='y', color={'field': annee, 'transform': color_mapper}, alpha=0.7, source=geo_source)

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




