# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import seaborn as sns
import self
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


#Inicialización temporal como dataframes de trafico_entrenamiento y trafico_datos_test
#trafico_entrenamiento = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(1)]
#trafico_datos_test = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(1)]
prediction_date = '2019-12-04'
# trafico_entrenamiento = trafico_union_semestres[~(trafico_union_semestres['fecha'] >= prediction_date)]
# trafico_datos_test = trafico_union_semestres[~(trafico_union_semestres['fecha'] == prediction_date)]
trafico_entrenamiento = pd.DataFrame()
trafico_datos_test = pd.DataFrame()

# el DF temporal
df = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(1)]

colors = {
    'background': '#777777',
    'text': '#7FDBFF'
}

fig = px.line(df, x='fecha', y='intensidadMedia',color="hora", title='Ejemplo')

# Este se cargará luego con el tendrá la misma fecha que el resultado de la prediccion de Prophet
#fig = px.bar(trafico_datos_test, x="fecha", y="intensidadMedia", color="idTramo", barmode="group" )

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

#def make_layout(fig):
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
    # html.Div(html.Button(id='submit-nuevabusqueda', type='submit', children='Nueva Busqueda'), id='nuevabusqueda', ),
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
            html.Br(), html.Br(),
            html.Div(id='output_div')
        ]),


    ], style={'padding': 10, 'flex': 4, 'color': colors['text']}),

    html.Div([
    dcc.Graph(
        id='graph-Dash-1',
        figure=fig,
    ),
    ], id='ejemploGraph',),

    html.Div( "", id='datosOriginales',),
    html.Div("", id='graphprophet', ),

    html.Div([html.Label("Comparar 1 fecha de 2019-01-06 hasta 2019-12-04 de la predicción con los datos"),
                dcc.Input(id='comparar', value='', type='text'),
                html.Button(id='submit-comparar', type='submit', children='Comparar'),
              ], style={'color': 'black', 'background-color': '#f5f5f5'}),
    html.Br(),
    html.Div(id='salidacomparar'),
    html.Div(html.Label("Sacar un gráfico con prophet"),id="graph-prophet-original",),

])

#This method is called by the method that does the prediction when the prediction date is introduced
def split_data_by_precdiction_date(predictiondate):
    if predictiondate != '':  # predictiondate es la fecha a predecir. Entrenamiento es el dataframe de entramiento
        prediction_date = predictiondate
        trafico_entrenamiento = trafico_union_semestres[~(trafico_union_semestres['fecha'] >= prediction_date)]
        trafico_datos_test = trafico_union_semestres[~(trafico_union_semestres['fecha'] == prediction_date)]
        #trafico_datos_test contiene el valor real de la predicción para comparar
        return trafico_entrenamiento, trafico_datos_test

#compares actual and predicted data for the same date, to see the precision accuracy
@app.callback(
    Output('salidacomparar', component_property='children'),
    Input('submit-comparar', 'n_clicks'),
    [State('comparar', 'value')],
    [State('horapredecir', 'value')],
    [State('fecha-diaConcreto', 'value')],
    [State('dropdown-descricion', 'value')],
)
def update_output_div(clicks, fecha_comparar,   hora, fechaDiaConcreto, tramodescripcion):
    if clicks is not None:
        numeroDeDias = 1 #Numero dias a predecir 1
        idtramo = ''
#        trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
        df = trafico_entrenamiento
        if tramodescripcion != '':
            idtramo = giveme_idtramo(tramodescripcion)
        if idtramo != '':  #df contiene un sub dataframe con solo los datos de ese tramo
            df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == int(idtramo)]
            df = df.loc[df['hora'] == int(hora)]

            df.set_index('fecha', inplace=True)
            ts = pd.DataFrame({'ds': df.index, 'y': df.intensidadMedia})
            prophet, forecast = prophet_plot(ts, numeroDeDias)
            dato = forecast.loc[
                forecast['ds'] == fecha_comparar, 'yhat']

            df2 = trafico_datos_test.loc[trafico_datos_test['idTramo'] == int(idtramo)]
            df2 = df2.loc[df2['hora'] == int(hora)]
            df2.head()
            dato2 = df2.loc[df2['fecha'] == fecha_comparar, 'intensidadMedia']

            return f'Predecido: {float(dato)} - dato real: {float(dato2)}'

# show the numeric id of the street
@app.callback(
    Output('dd-output-container', 'children'),
    Input('dropdown-descricion', 'value')
)
def update_output(value):
    if value != 0 and value != '':
        return html.H4('id de tramo= ' + str(giveme_idtramo(value)))

# recibe el nombre de tramo y busca y devuelve el id númerico del tramo
def giveme_idtramo(nomtramo): # se busca la descripcion que es igual a value y se coge el idtramo y se devuelve
    if nomtramo != 0 and nomtramo != '' and trafico_union_semestres.loc[
        trafico_union_semestres['descripcion'] == nomtramo, 'idTramo'].unique().size > 0:
        idtramo = trafico_union_semestres.loc[
            trafico_union_semestres['descripcion'] == nomtramo, 'idTramo'].unique()[0]
        return idtramo

# Shows the real data for the time series for the street for all hours, not the prediction (candidate to disappear)
@app.callback(Output('ejemploGraph', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2_originales(clicks, fechaDiaConcreto, tramodes):
    if clicks is not None:
        print(clicks, fechaDiaConcreto)
        trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
        idtramo = ''
        df = trafico_entrenamiento
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_datos_test.loc[trafico_datos_test['idTramo'] == int(idtramo)]
            nombretramo = trafico_datos_test.loc[
                trafico_datos_test['idTramo'] == int(idtramo), 'descripcion'].unique()[0]

        return dcc.Graph(
            id='prophetfig',
            figure=px.line(df, x='fecha', y='intensidadMedia',color="hora", title='todas las horas')
        )

# Shows the test data, real data not predicted for only the specified hour.
@app.callback(Output('datosOriginales', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2_originales(clicks, hora, fechaDiaConcreto, tramodes):
    if clicks is not None:
#        print(clicks, fechaDiaConcreto)
#        trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
        idtramo = ''
        df = trafico_entrenamiento
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_datos_test.loc[trafico_datos_test['idTramo'] == int(idtramo)]
            nombretramo = trafico_datos_test.loc[
                trafico_datos_test['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
            df = df.loc[df['hora'] == int(hora)]

        return dcc.Graph(
            id='prophetfig',

            figure=px.line(df, x='fecha', y='intensidadMedia', color="hora", title=nombretramo + ' ' + "Datos originales")
        )

# Shows the time series predicted data in a line graph using Dash line function
@app.callback(Output('graphprophet', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2(clicks, hora, fechaDiaConcreto, tramodes):
    if clicks is not None:
#        print(clicks, fechaDiaConcreto)
        numeroDeDias = numOfDays('2019-01-06', fechaDiaConcreto)
#        trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
        idtramo = ''
        df = trafico_entrenamiento
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
#            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == int(idtramo)]
            nombretramo= trafico_entrenamiento.loc[
                trafico_entrenamiento['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
#            df = df.loc[df['fecha'].dt.day == int(diaConcreto_value)]
            df = df.loc[df['hora'] == int(hora)]

        df.set_index('fecha', inplace=True)
        ts = pd.DataFrame({'ds': df.index, 'y': df.intensidadMedia})
        prophet, forecast = prophet_plot(ts, numeroDeDias)

        media = sum(df.intensidadMedia) / len(df.intensidadMedia)
        varianza_originales = sum((l-media)**2 for l in df.intensidadMedia) / len(df.intensidadMedia)
        st_dev_originales = math.sqrt(varianza_originales)
        varianza_predecidos = sum((l-media)**2 for l in forecast.yhat) / len(forecast.yhat)
        st_dev_predecidos = math.sqrt(varianza_predecidos)


        return html.Div([ dcc.Graph(
            id='prophetfig',
            figure=px.line(forecast, x='ds', y='yhat', title='Prophet Predicciones')
        ), "Desviación típica Originales = " + str(st_dev_originales) + " - Desviación típica Predecidos respecto Media de Originales = " + str(st_dev_predecidos)])

#Shows the prophet prediction in the Prophet original graphic using plotly
@app.callback(Output('graph-prophet-original', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2(clicks, hora, fechaDiaConcreto, tramodes):
    if clicks is not None:
#        print(clicks, fechaDiaConcreto)
        numeroDeDias = numOfDays('2019-01-06', fechaDiaConcreto)
        trafico_entrenamiento, trafico_datos_test = split_data_by_precdiction_date(fechaDiaConcreto)
        idtramo = ''
        df = trafico_entrenamiento
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
#            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_entrenamiento.loc[trafico_entrenamiento['idTramo'] == int(idtramo)]
            nombretramo= trafico_entrenamiento.loc[
                trafico_entrenamiento['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
#            df = df.loc[df['fecha'].dt.day == int(diaConcreto_value)]
            df = df.loc[df['hora'] == int(hora)]

        df.set_index('fecha', inplace=True)
        ts = pd.DataFrame({'ds': df.index, 'y': df.intensidadMedia})
        prophet, forecast = prophet_plot(ts, numeroDeDias)

        return html.Div([ dcc.Graph(
            id='prophetfig2',
            figure=plot_plotly(prophet, forecast)
        ),])

# return the prophet object function and the forecast dataframe for the prediction
def prophet_plot(ts, numeroDeDias):
    prophet = Prophet()
    prophet.fit(ts)
    future = prophet.make_future_dataframe(periods=int(numeroDeDias))
    forecast = prophet.predict(future)
    #figure.savefig('output')
    return prophet, forecast

def numOfDays(date1, date2):
    format = '%Y-%m-%d'
    d1 = datetime.strptime(date1, format).date()
    d2 = datetime.strptime(date2, format).date()
    print(d1)
    print(d2)
    return (d2 - d1).days


if __name__ == '__main__':
    app.run_server(debug=True)
