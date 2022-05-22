# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output, State
from fbprophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import pandas as pd
import self as self
import seaborn as sns
from datetime import date, datetime

from pygments.lexers import go

sns.set(color_codes=True)

app = Dash(__name__)

trafico_union_semestres = pd.read_pickle('trafico_union_semestres.plk')
trafico_union_semestres.fecha = pd.to_datetime(trafico_union_semestres.fecha)
descripcion = trafico_union_semestres['descripcion'].unique()
idTramoDescripcion = trafico_union_semestres[['descripcion', 'idTramo']].drop_duplicates()
trafico_union_semestres_sin2019 = trafico_union_semestres[~(trafico_union_semestres['fecha'] > '2019-01-06')]
trafico_union_semestres_solo2019 = trafico_union_semestres[~(trafico_union_semestres['fecha'] <= '2019-01-06')]

#trafico_union_semestres_solo2019['intensidadMedia'] = trafico_union_semestres_solo2019['intensidad']['mean']
intensidadMedia = trafico_union_semestres_solo2019['intensidad']['mean']
trafico_union_semestres_solo2019['intensidadMedia'] = intensidadMedia

trafico_union_semestres_solo2019.drop('intensidad', level=0, axis=1, inplace=True)
trafico_union_semestres_solo2019.drop('ocupacion', level=0, axis=1, inplace=True)
trafico_union_semestres_solo2019.drop('velocidad', level=0, axis=1, inplace=True)
#trafico_union_semestres_solo2019.drop(['intensidad']['count'], axis = 1)
#print(trafico_union_semestres_solo2019.head())

# el DF temporal
df = trafico_union_semestres_solo2019.loc[trafico_union_semestres_solo2019['idTramo'] == int(1)]

dataframeprophet = pd.DataFrame()

colors = {
    'background': '#777777',
    'text': '#7FDBFF'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })
# Load a small dataframe as e.g. and when the prophet loads the result update this dataframe also
#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
#fig = px.bar(df, x="fecha", y="intensidadMedia", color="intensidadMedia", barmode="group" )
fig = px.line(df, x='fecha', y='intensidadMedia',color="hora", title='Ejemplo')

# Este se cargará luego con el tendrá la misma fecha que el resultado de la prediccion de Prophet
#fig = px.bar(trafico_union_semestres_solo2019, x="fecha", y="intensidadMedia", color="idTramo", barmode="group" )

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
    html.Div( "", id='graphprophet',),
    html.Div( "", id='datosOriginales',),

    html.Div([html.Label("Comparar 1 fecha de 2019-01-06 hasta 2019-12-04 de la predicción con los datos"),
                dcc.Input(id='comparar', value='', type='text'),
                html.Button(id='submit-comparar', type='submit', children='Comparar'),
              ], style={'color': 'black', 'background-color': '#f5f5f5'}),
    html.Br(),
    html.Div(id='salidacomparar')

])


@app.callback(
    Output('salidacomparar', component_property='children'),
    Input('submit-comparar', 'n_clicks'),
    [State('comparar', 'value')],
    [State('horapredecir', 'value')],
    [State('fecha-diaConcreto', 'value')],
    [State('dropdown-descricion', 'value')],
)
def update_output_div(clicks, input_value,   hora, fechaDiaConcreto, tramodes):
    if clicks is not None:
        numeroDeDias = numOfDays('2019-01-06', fechaDiaConcreto)
        idtramo = ''
        df = trafico_union_semestres_sin2019
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
            df = trafico_union_semestres_sin2019.loc[trafico_union_semestres_sin2019['idTramo'] == int(idtramo)]
            nombretramo = trafico_union_semestres_sin2019.loc[
                trafico_union_semestres_sin2019['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
            df = df.loc[df['hora'] == int(hora)]
            df['media'] = df['intensidad']['mean']

            df.set_index('fecha', inplace=True)
            ts = pd.DataFrame({'ds': df.index, 'y': df.media})
            prophet, forecast = prophet_plot(ts, numeroDeDias)
            dato = forecast.loc[
                forecast['ds'] == input_value, 'yhat']

            df2 = trafico_union_semestres_solo2019.loc[trafico_union_semestres_solo2019['idTramo'] == int(idtramo)]
            df2 = df2.loc[df2['hora'] == int(hora)]
            df2['media'] = df2['intensidadMedia']
            df2.head()
            dato2 = df2.loc[df2['fecha'] == input_value, 'media']

            return f'Predecido: {float(dato)} - dato real: {float(dato2)}'


@app.callback(
    Output('dd-output-container', 'children'),
    Input('dropdown-descricion', 'value')
)
def update_output(value):
    if value != 0 and value != '':
        return html.H4('id de tramo= ' + str(giveme_idtramo(value)))


def giveme_idtramo(value): # se busca la descripcion que es igual a value y se coge el idtramo
    if value != 0 and value != '' and trafico_union_semestres_sin2019.loc[
        trafico_union_semestres_sin2019['descripcion'] == value, 'idTramo'].unique().size > 0:
        idtramo = trafico_union_semestres_sin2019.loc[
            trafico_union_semestres_sin2019['descripcion'] == value, 'idTramo'].unique()[0]
        return idtramo

@app.callback(Output('ejemploGraph', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2_originales(clicks, fechaDiaConcreto, tramodes):
    if clicks is not None:
        print(clicks, fechaDiaConcreto)
        idtramo = ''
        df = trafico_union_semestres_sin2019
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_union_semestres_solo2019.loc[trafico_union_semestres_solo2019['idTramo'] == int(idtramo)]
            nombretramo = trafico_union_semestres_solo2019.loc[
                trafico_union_semestres_solo2019['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
            df['media'] = df['intensidadMedia'] # solo calculo la media de intensidad

        return dcc.Graph(
            id='prophetfig',
            figure=px.line(df, x='fecha', y='intensidadMedia',color="hora", title='todas las horas')
        )

@app.callback(Output('datosOriginales', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2_originales(clicks, hora, fechaDiaConcreto, tramodes):
    if clicks is not None:
        print(clicks, fechaDiaConcreto)
        idtramo = ''
        df = trafico_union_semestres_sin2019
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_union_semestres_solo2019.loc[trafico_union_semestres_solo2019['idTramo'] == int(idtramo)]
            nombretramo = trafico_union_semestres_solo2019.loc[
                trafico_union_semestres_solo2019['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
            df = df.loc[df['hora'] == int(hora)]
            df['media'] = df['intensidadMedia'] # solo calculo la media de intensidad

#        return px.line(forecast, x='ds', y='yhat', title='Predicciones')
        return dcc.Graph(
            id='prophetfig',
#            figure=prophet.plot(forecast)
            figure=px.line(df, x='fecha', y='media', color="hora", title=nombretramo + ' ' + "Datos originales")
#            figure=px.line(forecast, x='ds', y='yhat', color='hour', title='Prophet')
        )

@app.callback(Output('graphprophet', 'children'),
#@app.callback(Output('prophetfig', 'figure'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('horapredecir', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2(clicks, hora, fechaDiaConcreto, tramodes):
    if clicks is not None:
        print(clicks, fechaDiaConcreto)
        numeroDeDias = numOfDays('2019-01-06', fechaDiaConcreto)
        idtramo = ''
        df = trafico_union_semestres_sin2019
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
            print(idtramo) # Me quedo con todas las filas que tengan el idtramo
            df = trafico_union_semestres_sin2019.loc[trafico_union_semestres_sin2019['idTramo'] == int(idtramo)]
            nombretramo= trafico_union_semestres_sin2019.loc[
                trafico_union_semestres_sin2019['idTramo'] == int(idtramo), 'descripcion'].unique()[0]
#            df = df.loc[df['fecha'].dt.day == int(diaConcreto_value)]
            df = df.loc[df['hora'] == int(hora)]
            df['media'] = df['intensidad']['mean'] # solo calculo la media de intensidad

        df.set_index('fecha', inplace=True)
        ts = pd.DataFrame({'ds': df.index, 'y': df.media})
        prophet, forecast = prophet_plot(ts, numeroDeDias)

        return html.Div([ dcc.Graph(
            id='prophetfig',
#            figure=prophet.plot(forecast)
#            fig=px.line(df, x='fecha', y='intensidadMedia', color="hora", title='Ejemplo')
            figure=px.line(forecast, x='ds', y='yhat', title='Prophet Predicciones')
        ),])


def prophet_plot(ts, numeroDeDias):
    prophet = Prophet()
    prophet.fit(ts)
    future = prophet.make_future_dataframe(periods=int(numeroDeDias))
    forecast = prophet.predict(future)
#    fig = prophet.plot(forecast)
    #    fig.show()
    #     fig.update_layout(
    #         plot_bgcolor=colors['background'],
    #         paper_bgcolor=colors['background'],
    #         font_color=colors['text']
    #     )
    #figure.savefig('output')
    return prophet, forecast




def numOfDays(date1, date2):
    format = '%Y-%m-%d'
    d1 = datetime.strptime(date1, format).date()
    d2 = datetime.strptime(date2, format).date()
    print(d1)
    print(d2)
    return (d2 - d1).days


#make_layout(fig)


if __name__ == '__main__':
    app.run_server(debug=True)
