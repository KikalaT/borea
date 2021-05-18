#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, CARTODBPOSITRON
from bokeh.models.tools import WheelZoomTool, BoxSelectTool
from bokeh.models import ColumnDataSource, LinearColorMapper, CustomJS
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.layouts import grid

import streamlit as st

# page configuration
st.set_page_config(
page_title="BoRea Xplore v1.0",
layout="wide",
)

# title
st.title('Sprattus sprattus (1850-2017)')

"""
### Source de la méthode d'analyse des données :
[consultable ici](https://github.com/KikalaT/borea/blob/main/sprat.ipynb)
"""
# ~ csv_file = st.file_uploader('Téléversez votre fichier CSV')

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def preprocessing():

	df_data = pd.read_csv('https://www.jazzreal.org/static/export_data_sprat_data.csv', index_col=0)
	k = 6378137
	df_data["x"] = df_data['Longitude'] * (k * np.pi / 180.0)
	df_data["y"] = np.log(np.tan((90 + df_data['Latitude']) * np.pi / 360.0)) * k
	return df_data
	
df_ = preprocessing()

"""
### Aperçu des données modélisées :
"""

st.write(df_.head())

"""
### Géolocalisation des données :
"""

annee = st.selectbox('Année',df_.columns.unique()[2:])

# création des sources de travail
df_years = pd.DataFrame(df_.loc[:,annee])
df_gps = pd.DataFrame(df_.loc[:,['x','y']]) 
df_show = pd.concat([df_gps,df_years], axis=1)
df_mean = pd.DataFrame({'mean':df_.loc[:,'1850':'2017'].mean(axis=1)})

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

# create datasources
s2 = ColumnDataSource(data={'index':[],'mean_value':[]})
s3 = ColumnDataSource(data=df_mean)

# create dynamic table of selected points
columns = [
	TableColumn(field='index', title="Index"),
	TableColumn(field="mean_value", title="Probabilité moyenne"),
]

table = DataTable(
	source=s2,
	columns=columns,
	width=400,
	height=600,
	sortable=True,
	selectable=True,
	editable=True,
)

s1.selected.js_on_change(
	"indices",
	CustomJS(
		args=dict(s3=s3, s2=s2, table=table),
		code="""
		var inds = cb_obj.indices;
		var d2 = s2.data;
		var d3 = s3.data;
		
		d2['index'] = inds
		d2['mean_value'] = []
		for (var i = 0; i < inds.length; i++) {
            d2['mean_value'].push(d3['mean'][inds[i]])
        }
		s2.change.emit();
		table.change.emit();
		"""
		)
		)

layout = grid([p, table], ncols=2)

st.bokeh_chart(layout)
