# -*- coding: utf-8 -*-
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
import plotly.express as px
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
from prophet.plot import plot_plotly

# Modeling and Forecasting
# ==============================================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Warnings configuration
# ==============================================================================
import warnings


import math
import time
from datetime import datetime


import dash
import seaborn as sns
from dash import Dash, dcc, html, Input, Output, State
from prophet import Prophet


# Configurations

sns.set(color_codes=True)

sns.set(color_codes=True)

app = Dash(__name__)

# trafico_union_semestres = pd.read_pickle('trafico_union_semestres.plk')
trafico_union_semestres = pd.read_pickle('tramosFechasCompletas.plk')

# 'mean' está en un segundo nivel dentro del dataframe y de intensidad, creo la columna 'intensidadMedia'
intensidadMedia = trafico_union_semestres['intensidad']['mean']
# borro las tres variables con 2 nivels, solo voy a utilizar la intensidad en este caso
trafico_union_semestres.drop('intensidad', level=0, axis=1, inplace=True)
trafico_union_semestres.drop('ocupacion', level=0, axis=1, inplace=True)
trafico_union_semestres.drop('velocidad', level=0, axis=1, inplace=True)
trafico_union_semestres.fecha = pd.to_datetime(trafico_union_semestres.fecha)

# meto la columna intensidadMedia por la que había antes de 2 nivels intensidad con media y count dentro.
trafico_union_semestres['intensidadMedia'] = intensidadMedia
# fechamax = trafico_union_semestres['fecha'] < pd.to_datetime('2019-12-03')
# print('FECHA MAX',fechamax)
# print('FECHA MAX head',trafico_union_semestres.head())
# print('FECHA MAX tail',trafico_union_semestres.tail())


# lleganAFecha = trafico_union_semestres.loc[fechamax].groupby(['idTramo', 'descripcion', 'hora','fecha','intensidadMedia'])['fecha'].idxmax()
# trafico_union_semestres.loc[lleganAFecha]

descripcion = trafico_union_semestres['descripcion'].unique()
idTramoDescripcion = trafico_union_semestres[['descripcion', 'idTramo']].drop_duplicates()

prediction_date = '2019-12-04'

trafico_entrenamiento = pd.DataFrame()
trafico_datos_test = pd.DataFrame()
column_names = ["Nombre_de_la_Calle_", "Fecha", "Hora", "Prediccion_Prophet", "Valor_Real", "Prediccion_Skforecast",
                "MSERROR_Prophet", "MSERROR_Skforecast"]
RESULTS = pd.DataFrame(columns=column_names)
print(RESULTS.head())
forecast = pd.DataFrame(data=None)


def dame_results():
    return RESULTS


# el DF temporal
df = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(1)]

colors = {
    'background': '#777777',
    'text': '#7FDBFF'
}

fig = px.line(df, x='fecha', y='intensidadMedia', color="hora", title='Ejemplo')

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(
    style={
        'backgroundColor': colors['background'],
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif'
    },
    children=[
        html.H1(
            children='Predicción de Tráfico en Valencia con Prophet, Dash y Skforecast',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'marginBottom': '40px'
            }
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label(
                            'Seleccione una descripción de tramo para predecir:',
                            style={'fontWeight': 'bold'}
                        ),
                        dcc.Dropdown(
                            descripcion,
                            'Pintor Sorolla, Nª 1',
                            id='dropdown-descricion',
                            style={'color': 'black'}
                        ),
                        html.Div(id='dd-output-container', style={'marginTop': '10px'}),
                    ],
                    style={
                        'width': '45%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'marginRight': '5%',
                        'color': colors['text']
                    }
                ),
                html.Div(
                    children=[
                        html.Label(
                            'Hora a predecir (0 para las 12 noche, de 0 a 23):',
                            style={'fontWeight': 'bold'}
                        ),
                        dcc.Input(
                            id='horapredecir',
                            value='',
                            type='text',
                            style={'width': '100%', 'marginBottom': '10px'}
                        ),
                        html.Label(
                            'Hasta que fecha predecir (formato: yyyy-mm-dd):',
                            style={'fontWeight': 'bold'}
                        ),
                        dcc.Input(
                            id='fecha-diaConcreto',
                            value='',
                            type='text',
                            style={'width': '100%', 'marginBottom': '10px'}
                        ),
                        html.Button(
                            id='submit-diaConcreto',
                            type='submit',
                            children='Predecir',
                            style={
                                'backgroundColor': '#4CAF50',
                                'color': 'white',
                                'padding': '10px 20px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'marginTop': '10px'
                            }
                        ),
                        html.Button(
                            id='submit-reset',
                            type='submit',
                            children='Reset Tabla de datos',
                            style={
                                'backgroundColor': '#f44336',
                                'color': 'white',
                                'padding': '10px 20px',
                                'border': 'none',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'marginLeft': '10px',
                                'marginTop': '10px'
                            }
                        ),
                        html.Div(id='output_div', style={'marginTop': '20px'})
                    ],
                    style={'width': '45%', 'display': 'inline-block', 'color': colors['text']}
                ),
            ],
            style={'marginBottom': '40px'}
        ),
        html.Div(
            children=[
                html.Label("Predicción Dash", style={'fontWeight': 'bold'}),
                dcc.Graph(
                    id='graph-Dash-1',
                    figure=fig,
                    style={'marginBottom': '40px'}
                ),
            ],
            id='prediccionDash',
        ),
        html.Div(
            children=[
                html.Label("Predicción Skforecast", style={'fontWeight': 'bold'}),
                dcc.Graph(
                    id='graph-Scikit-1',
                    figure=fig,
                    style={'marginBottom': '40px'}
                ),
            ],
            id='prediccionSkforecast',
        ),
        html.Div(
            html.Div(
                html.Label("Sacar un gráfico con Prophet (plot_plotly)", style={'fontWeight': 'bold'}),
                id="prediccion-plot_plotly-prophet",
            ),
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'marginBottom': '40px'
            }
        ),
        html.Div(
            children=[
                html.H4(
                    children='Comparación de resultados predichos y reales',
                    style={'textAlign': 'center', 'marginBottom': '20px'}
                ),
            ],
            id='tablaComparación',
        ),
    ]
)


# This method is called by the method that does the prediction when the prediction date is introduced
def dividir_datos_entrenamiento_y_test(predictiondate):
    if predictiondate != '':  # predictiondate es la fecha a predecir. Entrenamiento es el dataframe de entramiento
        prediction_date = predictiondate
        trafico_entrenamiento = trafico_union_semestres[~(trafico_union_semestres['fecha'] >= prediction_date)]
        trafico_datos_test = trafico_union_semestres[~(trafico_union_semestres['fecha'] == prediction_date)]
        # trafico_datos_test contiene el valor real de la predicción para comparar
        return trafico_entrenamiento, trafico_datos_test


# show the numeric id of the street
@app.callback(
    Output('dd-output-container', 'children'),
    Input('dropdown-descricion', 'value')
)
def update_output(value):
    if value != 0 and value != '':
        return html.H4('id de tramo= ' + str(giveme_idtramo(value)))


# recibe el nombre de tramo y busca y devuelve el id númerico del tramo
def giveme_idtramo(nomtramo):  # se busca la descripcion que es igual a value y se coge el idtramo y se devuelve
    if nomtramo != 0 and nomtramo != '' and trafico_union_semestres.loc[
        trafico_union_semestres['descripcion'] == nomtramo, 'idTramo'].unique().size > 0:
        idtramo = trafico_union_semestres.loc[
            trafico_union_semestres['descripcion'] == nomtramo, 'idTramo'].unique()[0]
        return idtramo


# Shows the real data for the time series for the street for all hours, not the prediction (candidate to disappear)
@app.callback(Output('prediccionDash', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              [State('horapredecir', 'value')], )
def prediccion_dash(clicks, fechaDiaConcreto, tramodes, hora):
    print("prediccion_dash")
    if clicks is not None and hora and fechaDiaConcreto and tramodes:
        print(clicks, fechaDiaConcreto)
        idtramo = ''
        numeroDeDias = 1
        print('Descripción de tramo en primer gráfico prediccionDash == ', tramodes)
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
            print('idtramo en gráfico 1 == ', idtramo)
        if idtramo != '':
            ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)
            prophet, forecast = devolver_prediccion_prophet(ts, numeroDeDias)

        # Primer gráfico
        return dcc.Graph(
            id='prophetfig',
            figure=px.line(forecast, x='ds', y='yhat', title='Gráfico Dash para Prophet'))


def datosEntrenamientoYTestSkforecast(idtramo, hora, ts, fechaDiaConcreto):
    trafico_entrenamiento, trafico_datos_test = dividir_datos_entrenamiento_y_test(fechaDiaConcreto)
    dftest = trafico_datos_test.loc[trafico_datos_test['idTramo'] == idtramo]
    dftest = dftest[dftest['hora'] == int(hora)]
    dftest.set_index('fecha', inplace=True)
    dftest = pd.DataFrame({'date': dftest.index, 'y': dftest.intensidadMedia})
    data_train = pd.DataFrame(columns=["date", "y"])
    data_train = ts.rename(columns={'ds': 'date'})  # en ts falta el dia de la prediccion
    data_train['date'] = pd.to_datetime(data_train['date'], format='%Y/%m/%d')
    data_train = data_train.set_index('date')
    data_train = data_train.sort_index()
    return dftest, data_train


@app.callback(Output('prediccionSkforecast', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              [State('horapredecir', 'value')],
              )
def prediccion_Skforecast(clicks, fechaDiaConcreto, tramodes, hora):
    print("prediccion_dash")
    if clicks is not None and hora and fechaDiaConcreto and tramodes:
        print(clicks, fechaDiaConcreto)
        idtramo = ''
        numeroDeDias = 1
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
            print('idtramo en gráfico 1 == ', idtramo)
        if idtramo != '':
            ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)

            # Skforecast prediction
            dftest, data_train = datosEntrenamientoYTestSkforecast(idtramo, hora, ts, fechaDiaConcreto)

            steps = 1  # numero de días a predecir por Skforecast
            #            lags = data_train.shape[0]  # el numero de elementos utilizados para entrenar y generar la predección
            forecaster = devolver_forecaster()  # crea el regresor con random forest y con el regresor crea el forecaster
            data_train = data_train.fillna(data_train['y'].mean())
            forecaster.fit(y=data_train['y'])
            predictionsSkforecast = forecaster.predict(steps=steps)
            predictionSK = pd.DataFrame({'date': pd.to_datetime(fechaDiaConcreto), 'y': predictionsSkforecast.values})
            data_train['date'] = pd.to_datetime(data_train.index, format='%Y/%m/%d')
            data_train = pd.concat([data_train, predictionSK], ignore_index=True)
            print("DATA_TRAIN LINEA 277", data_train.tail())

        # Primer gráfico
        return dcc.Graph(
            id='prophetfig',
            #            figure=px.line(df, x='fecha', y='intensidadMedia',color="hora", title='todas las horas'),
            figure=px.line(data_train, x='date', y='y', title='Skforecast Predicciones'))


def devolver_forecaster():
    regressor = RandomForestRegressor(max_depth=3, n_estimators=100,
                                      random_state=123)  # maxima profundida de los arboles 3, numero de arboles 100
    forecaster = ForecasterAutoreg(
        regressor=regressor,
        lags=36  # time window for reading data 36 months
    )
    return forecaster


# Shows the prophet prediction in the Prophet original graphic using plotly
@app.callback(Output('prediccion-plot_plotly-prophet', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def grafico_real_prophet_plotly(clicks, hora, fechaDiaConcreto, tramodes):
    print(grafico_real_prophet_plotly)
    if clicks is not None and hora and fechaDiaConcreto and tramodes:
        numeroDeDias = 1
        idtramo = ''

        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
            print('idtramo en gráfico 4 == ', idtramo)
        if idtramo != '':
            ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)
            prophet, forecast = devolver_prediccion_prophet(ts, numeroDeDias)

            return html.Div([dcc.Graph(
                id='prophetfig2',
                figure=plot_plotly(prophet, forecast, figsize=(1500, 700))
            ), ])


# return the prophet object function and the forecast dataframe for the prediction
def devolver_prediccion_prophet(ts, numeroDeDias):
    # vacaciones = pd.DataFrame({
    #     'holiday': 'semanaSanta',
    #     'ds': pd.to_datetime(['2017-04-09', '2017-04-10', '2017-04-11', '2017-04-12', '2017-04-13', '2017-04-14',
    #                           '2017-04-15', '2017-04-16', '2018-03-25', '2018-03-26', '2018-03-27', '2018-03-28',
    #                           '2018-03-29', '2018-03-30', '2018-03-31', '2018-04-01', '2019-04-14', '2018-04-15',
    #                           '2019-04-16', '2018-04-17', '2019-04-18', '2019-04-19', '2019-04-20', '2019-04-21']),
    #     'lower_window': 0,
    #     'upper_window': 1,
    # })
    # holidays=vacaciones,

    global forecast  # changepoint_range es el range de los puntos de cambio, se utiliza el 100% = 1 de los datos para identificar los puntos de cambio
    prophet = Prophet(changepoint_range=1, changepoint_prior_scale=0.1, daily_seasonality=True, weekly_seasonality=True,
                      yearly_seasonality=True)
    prophet.fit(
        ts)  # changepoint_prior_scale 0.05 permite la detección automática de los puntos de cambio, aumentarlo para que detecte más puntos de cambio
    future = prophet.make_future_dataframe(periods=int(numeroDeDias))
    forecast = prophet.predict(future)
    # figure.savefig('output')
    return prophet, forecast


def devolverintensidadMediaReal(idtramo, fechaDiaConcreto, hora):
    df_reales = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(idtramo)]
    df_reales = df_reales.loc[df_reales['fecha'] == fechaDiaConcreto]
    intensidadMediaReal = df_reales.loc[df_reales['hora'] == int(hora), 'intensidadMedia'].values[0]
    return intensidadMediaReal


@app.callback(Output('tablaComparación', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [Input('submit-reset', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              [State('horapredecir', 'value')],
              )
def mostrar_tabla_de_datos(clicks, clicks2, fechaDiaConcreto, tramodes, hora):
    button_id = dash.ctx.triggered_id if not None else 'No clicks yet'

    global RESULTS
    if clicks or clicks2 is not None and hora and fechaDiaConcreto and tramodes:
        if button_id == 'submit-reset':
            RESULTS.empty
            RESULTS = dame_results()
            RESULTS.empty
            RESULTS = pd.DataFrame(columns=column_names)
        else:  # 'submit-diaConcreto'
            idtramo = ''
            numeroDeDias = 1
            if tramodes != '':
                idtramo = giveme_idtramo(tramodes)
            if idtramo != '':

                ts = devolver_serie_temporal(hora, fechaDiaConcreto, tramodes)
                prophet, forecast = devolver_prediccion_prophet(ts, numeroDeDias)

                intensidadMediaPrediccionProphet = forecast.loc[forecast['ds'] == fechaDiaConcreto, 'yhat'].values[0]
                intensidadMediaReal = devolverintensidadMediaReal(idtramo, fechaDiaConcreto, hora)

                # Skforecast prediction
                dftest, data_train = datosEntrenamientoYTestSkforecast(idtramo, hora, ts, fechaDiaConcreto)

                steps = 1  # numero de días a predecir por Skforecast
                data_test = trafico_datos_test
                data_train = data_train.fillna(data_train[
                                                   'y'].mean())  # n_estimators: número de árboles incluidos en el modelo. random_state: semilla para que los resultados sean reproducibles
                forecaster = devolver_forecaster()
                forecaster.fit(y=data_train['y'])  # aquí se entrena el modelo de Skforecast
                predictionsSkforecast = forecaster.predict(steps=steps)

                prediccion_prophet = pd.DataFrame([[intensidadMediaPrediccionProphet]], columns=['Prediccion_Prophet'])
                if not RESULTS[
                    'Prediccion_Prophet'].empty:  # si no está vacío, se concatena con el dataframe de Predicciones de Prophet para calcular el error cuadrado medio
                    prophet_dataframe = pd.concat(
                        [RESULTS['Prediccion_Prophet'], prediccion_prophet['Prediccion_Prophet']], ignore_index=True,
                        axis=0)
                else:
                    prophet_dataframe = prediccion_prophet  # si está vacio, se le asigna el dataframe de predicciones de prophet, es el primer valor

                prediccion_skforecast = pd.DataFrame([[predictionsSkforecast.values[0]]], columns=[
                    'Prediccion_Skforecast'])  # Dataframe con la predicción de Skforecast
                skforecast_dataframe = pd.DataFrame(columns=[
                    'Prediccion_Skforecast'])  # to concatenate with the previous skforecast predictions for calculating errors
                if not RESULTS[
                    'Prediccion_Skforecast'].empty:  # si no está vacío, se concatena con el dataframe de Predicciones de Skforecast para calcular el error cuadrado medio
                    skforecast_dataframe = pd.concat(
                        [RESULTS['Prediccion_Skforecast'], prediccion_skforecast['Prediccion_Skforecast']],
                        ignore_index=True, axis=0)
                else:
                    skforecast_dataframe = prediccion_skforecast  # si está vacio, se le asigna el dataframe de predicciones de skforecast, es el primer valor

                valor_real = pd.DataFrame([[intensidadMediaReal]],
                                          columns=['Valor_Real'])  # Dataframe con el valor real
                real_dataframe = pd.DataFrame(
                    columns=['Valor_Real'])
                if not RESULTS[
                    'Valor_Real'].empty:  # si no está vacío, se concatena con el dataframe de valores reales para calcular el error cuadrado medio
                    real_dataframe = pd.concat([RESULTS['Valor_Real'], valor_real['Valor_Real']], ignore_index=True,
                                               axis=0)
                else:
                    real_dataframe = valor_real  # si está vacio, se le asigna el dataframe de valores reales, es el primer valor

                mserror_Prophet = math.sqrt(mean_squared_error(real_dataframe, prophet_dataframe))
                # skforecast_dataframe = skforecast_dataframe.fillna(skforecast_dataframe['Prediccion_Skforecast'].mean())
                # print("SKFORECAST_DATAFRAME ", skforecast_dataframe)
                mserror_Skforecast = math.sqrt(mean_squared_error(real_dataframe, skforecast_dataframe))

                dict = pd.DataFrame([[tramodes, fechaDiaConcreto, hora,
                                      float("{:.2f}".format(intensidadMediaPrediccionProphet)),
                                      float("{:.2f}".format(intensidadMediaReal)),
                                      float("{:.2f}".format(predictionsSkforecast.values[0])),
                                      float("{:.2f}".format(mserror_Prophet)),
                                      float("{:.2f}".format(mserror_Skforecast))]],
                                    columns=["Nombre_de_la_Calle_", "Fecha", "Hora", "Prediccion_Prophet", "Valor_Real",
                                             "Prediccion_Skforecast", "MSERROR_Prophet", "MSERROR_Skforecast"], )

                RESULTS = pd.concat([RESULTS, dict], ignore_index=True, axis=0)
            print('RESULTS =====######### ', RESULTS)
            return html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in RESULTS.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(RESULTS.iloc[i][col]) for col in RESULTS.columns
                    ]) for i in range(min(len(RESULTS), 100))
                ])
            ])


def devolver_serie_temporal(hora, fechaDiaConcreto, tramodes):
    if tramodes != '':
        idtramo = giveme_idtramo(tramodes)
        print('idtramo en gráfico 1 == ', idtramo)
    if idtramo != '':
        trafico_entrenamiento, trafico_datos_test = dividir_datos_entrenamiento_y_test(fechaDiaConcreto)
        # Me quedo con todas las filas que tengan el idtramo
        df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == idtramo]
        df = df.loc[df['hora'] == int(hora)]
        #        df.set_index('fecha', inplace=True)
        ts = pd.DataFrame({'ds': df['fecha'], 'y': df.intensidadMedia})
        return ts


if __name__ == '__main__':
    app.run_server(debug=True)
