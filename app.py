# -*- coding: utf-8 -*-
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import warnings
import math
import time
from datetime import datetime
import dash
import plotly.express as px
import seaborn as sns
from dash import Dash, dcc, html, Input, Output, State
from prophet import Prophet
from prophet.plot import plot_plotly

# Configurations
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
sns.set(color_codes=True)

# Initialize Dash app
app = Dash(__name__)

# Load data
trafico_union_semestres = pd.read_pickle('tramosFechasCompletas.plk')

# Data Preprocessing
intensidadMedia = trafico_union_semestres['intensidad']['mean']
trafico_union_semestres.drop(['intensidad', 'ocupacion', 'velocidad'], level=0, axis=1, inplace=True)
trafico_union_semestres['fecha'] = pd.to_datetime(trafico_union_semestres['fecha'])
trafico_union_semestres['intensidadMedia'] = intensidadMedia

descripcion = trafico_union_semestres['descripcion'].unique()
idTramoDescripcion = trafico_union_semestres[['descripcion', 'idTramo']].drop_duplicates()

# Initialize global variables
prediction_date = '2019-12-04'
column_names = [
    "Nombre_de_la_Calle_", "Fecha", "Hora", "Prediccion_Prophet", 
    "Valor_Real", "Prediccion_Skforecast", "MSERROR_Prophet", "MSERROR_Skforecast"
]
RESULTS = pd.DataFrame(columns=column_names)
forecast = pd.DataFrame()

def dame_results():
    return RESULTS

# Create initial plot
df = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == 1]
colors = {'background': '#777777', 'text': '#7FDBFF'}
fig = px.line(df, x='fecha', y='intensidadMedia', color="hora", title='Ejemplo')
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

# Dash layout
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Prediccion de Tráfico Valencia capital con Prophet, Dash y Skforecast.',
        style={'textAlign': 'center', 'color': colors['text']}
    ),
    html.Div(children=[
        html.Div([
            html.Label('Seleccione una descripción de tramo para predecir:'),
            dcc.Dropdown(descripcion, 'Pintor Sorolla, Nª 1', id='dropdown-descricion', style={'color': 'black'}),
            html.Div(id='dd-output-container'),
        ], style={'width': '50%', 'display': 'inline-block', 'color': colors['text']}),
        html.Br(),
        html.Div([
            "Hora a predecir formato(0 para las 12 noche, de 0 a 23): ",
            dcc.Input(id='horapredecir', value='', type='text'),
            "hasta que fecha predecir a partir de 2019-01-06 hasta 2019-12-04 formato (yyyy-mm-dd): ",
            dcc.Input(id='fecha-diaConcreto', value='', type='text'),
            html.Button(id='submit-diaConcreto', type='submit', children='Predecir'),
            html.Br(),
            html.Button(id='submit-reset', type='submit', children='Reset Tabla de datos'),
            html.Br(), html.Br(),
            html.Div(id='output_div')
        ]),
    ], style={'padding': 10, 'flex': 4, 'color': colors['text']}),
    html.Div([html.Label("prediccionDash"), dcc.Graph(id='graph-Dash-1', figure=fig)], id='prediccionDash'),
    html.Div([html.Label("prediccionSkforecast"), dcc.Graph(id='graph-Scikit-1', figure=fig)], id='prediccionSkforecast'),
    html.Br(), html.Br(),
    html.Div(
        html.Div(html.Label("Sacar un gráfico con prophet prediccion-plot_plotly-prophet"), id="prediccion-plot_plotly-prophet"),
        style={'display': 'flex', 'justify-content': 'center', 'display': 'inline-block'}
    ),
    html.Br(), html.Br(),
    html.Div([html.H4(children='Comparación de resultados predecidos y reales')], id='tablaComparación'),
])

# Helper functions and callbacks
def dividir_datos_entrenamiento_y_test(predictiondate):
    if predictiondate:
        prediction_date = predictiondate
        trafico_entrenamiento = trafico_union_semestres[~(trafico_union_semestres['fecha'] >= prediction_date)]
        trafico_datos_test = trafico_union_semestres[~(trafico_union_semestres['fecha'] == prediction_date)]
        return trafico_entrenamiento, trafico_datos_test

@app.callback(
    Output('dd-output-container', 'children'),
    Input('dropdown-descricion', 'value')
)
def update_output(value):
    if value:
        return html.H4(f'id de tramo= {giveme_idtramo(value)}')

def giveme_idtramo(nomtramo):
    if nomtramo and trafico_union_semestres.loc[trafico_union_semestres['descripcion'] == nomtramo, 'idTramo'].unique().size > 0:
        return trafico_union_semestres.loc[trafico_union_semestres['descripcion'] == nomtramo, 'idTramo'].unique()[0]

@app.callback(Output('prediccionDash', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              [State('horapredecir', 'value')])
def prediccion_dash(clicks, fechaDiaConcreto, tramodes, hora):
    if clicks and hora and fechaDiaConcreto and tramodes:
        idtramo = giveme_idtramo(tramodes)
        if idtramo:
            ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)
            prophet, forecast = devolver_prediccion_prophet(ts, 1)
            return dcc.Graph(id='prophetfig', figure=px.line(forecast, x='ds', y='yhat', title='Gráfico Dash para Prophet'))

def datosEntrenamientoYTestSkforecast(idtramo, hora, ts, fechaDiaConcreto):
    trafico_entrenamiento, trafico_datos_test = dividir_datos_entrenamiento_y_test(fechaDiaConcreto)
    dftest = trafico_datos_test.loc[trafico_datos_test['idTramo'] == idtramo]
    dftest = dftest[dftest['hora'] == int(hora)]
    dftest.set_index('fecha', inplace=True)
    dftest = pd.DataFrame({'date': dftest.index, 'y': dftest.intensidadMedia})
    data_train = ts.rename(columns={'ds': 'date'}).set_index('date').sort_index()
    return dftest, data_train

@app.callback(Output('prediccionSkforecast', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              [State('horapredecir', 'value')])
def prediccion_Skforecast(clicks, fechaDiaConcreto, tramodes, hora):
    if clicks and hora and fechaDiaConcreto and tramodes:
        idtramo = giveme_idtramo(tramodes)
        if idtramo:
            ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)
            dftest, data_train = datosEntrenamientoYTestSkforecast(idtramo, hora, ts, fechaDiaConcreto)
            forecaster = devolver_forecaster()
            data_train = data_train.fillna(data_train['y'].mean())
            forecaster.fit(y=data_train['y'])
            predictionsSkforecast = forecaster.predict(steps=1)
            predictionSK = pd.DataFrame({'date': pd.to_datetime(fechaDiaConcreto), 'y': predictionsSkforecast.values})
            data_train = data_train.concat(predictionSK)
            return dcc.Graph(id='prophetfig', figure=px.line(data_train, x='date', y='y', title='Skforecast Predicciones'))

def devolver_forecaster():
    regressor = RandomForestRegressor(max_depth=3, n_estimators=100, random_state=123)
    return ForecasterAutoreg(regressor=regressor, lags=36)

@app.callback(Output('prediccion-plot_plotly-prophet', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')])
def grafico_real_prophet_plotly(clicks, hora, fechaDiaConcreto, tramodes):
    if clicks and hora and fechaDiaConcreto and tramodes:
        idtramo = giveme_idtramo(tramodes)
        if idtramo:
            ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)
            prophet, forecast = devolver_prediccion_prophet(ts, 1)
            return html.Div([dcc.Graph(id='prophetfig2', figure=plot_plotly(prophet, forecast, figsize=(1500, 700)))])

def devolver_prediccion_prophet(ts, numeroDeDias):
    prophet = Prophet(changepoint_range=1, changepoint_prior_scale=0.1, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    prophet.fit(ts)
    future = prophet.make_future_dataframe(periods=numeroDeDias)
    forecast = prophet.predict(future)
    return prophet, forecast

def devolverintensidadMediaReal(idtramo, fechaDiaConcreto, hora):
    df_reales = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(idtramo)]
    df_reales = df_reales.loc[df_reales['fecha'] == fechaDiaConcreto]
    return df_reales.loc[df_reales['hora'] == int(hora), 'intensidadMedia'].values[0]

@app.callback(Output('tablaComparación', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [Input('submit-reset', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              [State('horapredecir', 'value')])
def mostrar_tabla_de_datos(clicks, clicks2, fechaDiaConcreto, tramodes, hora):
    global RESULTS
    if (clicks or clicks2) and hora and fechaDiaConcreto and tramodes:
        if dash.ctx.triggered_id == 'submit-reset':
            RESULTS = pd.DataFrame(columns=column_names)
        else:
            idtramo = giveme_idtramo(tramodes)
            if idtramo:
                ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)
                prophet, forecast = devolver_prediccion_prophet(ts, 1)
                intensidadMediaPrediccionProphet = forecast.loc[forecast['ds'] == fechaDiaConcreto, 'yhat'].values[0]
                intensidadMediaReal = devolverintensidadMediaReal(idtramo, fechaDiaConcreto, hora)

                dftest, data_train = datosEntrenamientoYTestSkforecast(idtramo, hora, ts, fechaDiaConcreto)
                forecaster = devolver_forecaster()
                forecaster.fit(y=data_train['y'])
                predictionsSkforecast = forecaster.predict(steps=1)

                mserror_Prophet = math.sqrt(mean_squared_error(RESULTS['Valor_Real'], RESULTS['Prediccion_Prophet']))
                mserror_Skforecast = math.sqrt(mean_squared_error(RESULTS['Valor_Real'], RESULTS['Prediccion_Skforecast']))

                dict_row = pd.DataFrame([[
                    tramodes, fechaDiaConcreto, hora, 
                    float("{:.2f}".format(intensidadMediaPrediccionProphet)), 
                    float("{:.2f}".format(intensidadMediaReal)),
                    float("{:.2f}".format(predictionsSkforecast.values[0])), 
                    float("{:.2f}".format(mserror_Prophet)), 
                    float("{:.2f}".format(mserror_Skforecast))
                ]], columns=column_names)

                RESULTS = pd.concat([RESULTS, dict_row], ignore_index=True)
            
            return html.Table([
                html.Thead(html.Tr([html.Th(col) for col in RESULTS.columns])),
                html.Tbody([
                    html.Tr([html.Td(RESULTS.iloc[i][col]) for col in RESULTS.columns]) 
                    for i in range(min(len(RESULTS), 100))
                ])
            ])

def devolver_serie_temporal(hora, fechaDiaConcreto, tramodes):
    idtramo = giveme_idtramo(tramodes)
    if idtramo:
        trafico_entrenamiento, trafico_datos_test = dividir_datos_entrenamiento_y_test(fechaDiaConcreto)
        df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == idtramo]
        df = df.loc[df['hora'] == int(hora)]
        ts = pd.DataFrame({'ds': df['fecha'], 'y': df.intensidadMedia})
        return ts

if __name__ == '__main__':
    app.run_server(debug=True)
