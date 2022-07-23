# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import math
import time
from datetime import datetime

import dash
import pandas as pd
import plotly.express as px
import seaborn as sns
from dash import Dash, dcc, html, Input, Output, State
from fbprophet import Prophet
from prophet.plot import plot_plotly

sns.set(color_codes=True)

app = Dash(__name__)

trafico_union_semestres = pd.read_pickle('trafico_union_semestres.plk')
# 'mean' está en un segundo nivel dentro del dataframe y de intensidad, creo la columna 'intensidadMedia'
intensidadMedia = trafico_union_semestres['intensidad']['mean']
# borro las tres variables con 2 nivels, solo voy a utilizar la intensidad en este caso
trafico_union_semestres.drop('intensidad', level=0, axis=1, inplace=True)
trafico_union_semestres.drop('ocupacion', level=0, axis=1, inplace=True)
trafico_union_semestres.drop('velocidad', level=0, axis=1, inplace=True)
trafico_union_semestres.fecha = pd.to_datetime(trafico_union_semestres.fecha)
# meto la columna intensidadMedia por la que había antes de 2 nivels intensidad con media y count dentro.
trafico_union_semestres['intensidadMedia'] = intensidadMedia

descripcion = trafico_union_semestres['descripcion'].unique()
idTramoDescripcion = trafico_union_semestres[['descripcion', 'idTramo']].drop_duplicates()

# Inicialización temporal como dataframes de trafico_entrenamiento y trafico_datos_test
# trafico_entrenamiento = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(1)]
# trafico_datos_test = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(1)]
prediction_date = '2019-12-04'
# trafico_entrenamiento = trafico_union_semestres[~(trafico_union_semestres['fecha'] >= prediction_date)]
# trafico_datos_test = trafico_union_semestres[~(trafico_union_semestres['fecha'] == prediction_date)]
trafico_entrenamiento = pd.DataFrame()
trafico_datos_test = pd.DataFrame()
column_names = ["Fecha", "Hora", "Predicción", "Valor Real"]

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

# Este se cargará luego con el tendrá la misma fecha que el resultado de la prediccion de Prophet
# fig = px.bar(trafico_datos_test, x="fecha", y="intensidadMedia", color="idTramo", barmode="group" )

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

    html.H1(
        children=['Prediccion de Tráfico Valencia capital con Prophet y Dash. ',
                  'Las fechas para predecir pudiendo comparar resultado van de 2019-01-06 hasta el 2019-12-04. ',
                  'Otras fechas posteriores no se podrá comparar el resultado con los datos pero se hará la predicción.'],
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
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


    html.Div([html.Label("prediccionDash"),
        dcc.Graph(
            id='graph-Dash-1',
            figure=fig,),
    ], id='prediccionDash'),

    html.Br(), html.Br(),

    html.Div(
    html.Div(html.Label("Sacar un gráfico con prophet prediccion-plot_plotly-prophet"),
             id="prediccion-plot_plotly-prophet", ),
    style={'display': 'flex', 'justify-content': 'center', 'display': 'inline-block'}),

    html.Br(), html.Br(),

    html.Div([
            html.H4(children='Comparación de resultados predecidos y reales'),
        ], id='tablaComparación', ),
])


# This method is called by the method that does the prediction when the prediction date is introduced
def split_data_by_precdiction_date(predictiondate):
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
              [State('horapredecir', 'value')],
              )
def prediccion_dash(clicks, fechaDiaConcreto, tramodes, hora):
    print(prediccion_dash)
    if clicks is not None:
        print(clicks, fechaDiaConcreto)
        idtramo = ''
        numeroDeDias = 1
        print('Descripción de tramo en primer gráfico prediccionDash == ', tramodes)
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
            print('idtramo en gráfico 1 == ', idtramo)
        if idtramo != '':
            trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
            # Me quedo con todas las filas que tengan el idtramo
            df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == idtramo]
            df = df.loc[df['hora'] == int(hora)]
            df.set_index('fecha', inplace=True)
            ts = pd.DataFrame({'ds': df.index, 'y': df.intensidadMedia})

            prophet, forecast = devolver_prediccion_prophet(ts, numeroDeDias)
            
        # Primer gráfico
        return dcc.Graph(
            id='prophetfig',
            #            figure=px.line(df, x='fecha', y='intensidadMedia',color="hora", title='todas las horas'),
            figure=px.line(forecast, x='ds', y='yhat', title='Prophet Predicciones')
        )

# Shows the prophet prediction in the Prophet original graphic using plotly
@app.callback(Output('prediccion-plot_plotly-prophet', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def grafico_real_prophet_plotly(clicks, hora, fechaDiaConcreto, tramodes):
    print(grafico_real_prophet_plotly)
    if clicks is not None:
        #        print(clicks, fechaDiaConcreto)
        #        numeroDeDias = numOfDays('2019-01-06', fechaDiaConcreto)
        numeroDeDias = 1
        trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
        idtramo = ''
        print('Descripción de tramo en cuarto gráfico prediccion-plot_plotly-prophet == ', tramodes)
        df = trafico_entrenamiento
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
            print('idtramo en gráfico 4 == ', idtramo)
        if idtramo != '':
            #            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == int(idtramo)]
            nombretramo = trafico_entrenamiento.loc[
                trafico_entrenamiento['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
            #            df = df.loc[df['fecha'].dt.day == int(diaConcreto_value)]
            df = df.loc[df['hora'] == int(hora)]

        df.set_index('fecha', inplace=True)
        ts = pd.DataFrame({'ds': df.index, 'y': df.intensidadMedia})
        prophet, forecast = devolver_prediccion_prophet(ts, numeroDeDias)

#        mostrar_tabla_de_datos(clicks, fechaDiaConcreto, tramodes, hora, forecast)

        return html.Div([dcc.Graph(
            id='prophetfig2',
            figure=plot_plotly(prophet, forecast)
        ), ])


# return the prophet object function and the forecast dataframe for the prediction
def devolver_prediccion_prophet(ts, numeroDeDias):
    global forecast
    prophet = Prophet()
    prophet.fit(ts)
    future = prophet.make_future_dataframe(periods=int(numeroDeDias))
    forecast = prophet.predict(future)
    # figure.savefig('output')
    return prophet, forecast


@app.callback(Output('tablaComparación', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [Input('submit-reset', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              [State('horapredecir', 'value')],
              )
def mostrar_tabla_de_datos(clicks, clicks2, fechaDiaConcreto, tramodes, hora):
    print("DASH CONTEXT IS: ======= ", dash.callback_context)
    button_id = dash.ctx.triggered_id if not None else 'No clicks yet'
    print("#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#",button_id)

    global RESULTS
    if clicks or clicks2 is not None:
        if button_id == 'submit-reset':
            RESULTS.empty
            RESULTS = dame_results()
            RESULTS.empty
            RESULTS = pd.DataFrame(columns=column_names)
        else:
            idtramo = ''
            numeroDeDias = 1
            if tramodes != '':
                idtramo = giveme_idtramo(tramodes)
            if idtramo != '':
                trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
                df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == int(idtramo)]
                df = df.loc[df['hora'] == int(hora)]
                df.set_index('fecha', inplace=True)
                ts = pd.DataFrame({'ds': df.index, 'y': df.intensidadMedia})

                prophet, forecast = devolver_prediccion_prophet(ts, numeroDeDias)

                intensidadMediaPrediccion = forecast.loc[forecast['ds'] == fechaDiaConcreto, 'yhat'].values[0]

                df_reales = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(idtramo)]
                df_reales = df_reales.loc[df_reales['fecha'] == fechaDiaConcreto]
                intensidadMediaReal = df_reales.loc[df_reales['hora'] == int(hora), 'intensidadMedia'].values[0]

                dict = pd.DataFrame([[fechaDiaConcreto, hora, float("{:.2f}".format(intensidadMediaPrediccion)), float("{:.2f}".format(intensidadMediaReal))]],
                                columns=["Fecha", "Hora", "Predicción", "Valor Real"])
                #    RESULTS = dame_results()
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

if __name__ == '__main__':
    app.run_server(debug=True)


#         media = sum(df.intensidadMedia) / len(df.intensidadMedia)
#         varianza_originales = sum((l-media)**2 for l in df.intensidadMedia) / len(df.intensidadMedia)
#         st_dev_originales = math.sqrt(varianza_originales)
#         varianza_predecidos = sum((l-media)**2 for l in forecast.yhat) / len(forecast.yhat)
#         st_dev_predecidos = math.sqrt(varianza_predecidos)
