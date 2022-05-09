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

colors = {
    'background': '#777777',
    'text': '#7FDBFF'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

    html.H1(
        children=['Prediccion de Tráfico Valencia capital',
                  'Las fechas para predecir pudiendo comparar resultado van de 2019-01-06 hasta el 2019-12-04. ',
                  'Otras fechas posteriores no se podrá comparar el resultado con los datos'],
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
        # html.Div([
        #     html.Label('Seleccione opcións de visualización:'),
        #     dcc.RadioItems(['Prediccion del dia entero concreto (e.g. día 25 o día 14 de todos los meses)',
        #                     'Predicción de algunas horas concretas de todos los días (e.g. de hora inicial 18 a final 19)',
        #                     'Predicción de un número de días desde la fecha inicial (e.g los próximos 30 o 365 días)',
        #                     'algúna/s horas concretas de un día en concreto (e.g. de 7 a 8 todos los días 15)'
        #                     ], "", id='options-radio', style={'display': 'block'}),
        # ]),
        # html.Br(),

        html.Div([
            html.Label('Que desea predecir: Intensidad(I), Ocupación(O), o Velocidad(V):'),
            dcc.Input(id='IntenoOcupaoVeloci', value='', type='text'),
        ]),

        html.Div([
            "dia entero concreto a predecir (e.g. dia 1 o el 2 o 3 o el 15 o el 30)DIA CONCRETO: ",
            dcc.Input(id='dia-diaConcreto', value='', type='text'),
            "hasta que fecha predecir dia concreto formato (yyyy-mm-dd)FECHA FINAL: ",
            dcc.Input(id='fecha-diaConcreto', value='', type='text'),
            html.Button(id='submit-diaConcreto', type='submit', children='Predecir'),
            html.Br(), html.Br(),
            html.Div(id='output_div')
        ]),
        html.Div([
            "Predicción de alguna/s horas concretas de todos los días (e.g. de hora inicial 18 a final 19): ",
            "Hora inicial formato(0 para las 12 noche, 1 para la una noche, 23 para las 11 noche)HORA INICIAL: ",
            dcc.Input(id='hora-inicial-horasConcretasTodos', value='', type='text'),
            "Hora final: formato(0 para las 12 noche, 1 para la una noche, 23 para las 11 noche)HORA FINAL:",
            dcc.Input(id='hora-final-horasConcretasTodos', value='', type='text'),
            "hasta que fecha predecir, fecha inicial final formato (yyyy-mm-dd)FECHA FINAL: ",
            dcc.Input(id='fecha-horasConcretasTodos', value='', type='text'),
            html.Button(id='submit-horasConcretasTodos', type='submit', children='Predecir'),
            html.Br(), html.Br(),
            html.Div(id='output_div2')

        ]),
        html.Div([
            "Predicción de alguna/s horas concretas de algún día concreto (e.g. día 19 de hora inicial 18 a final 19):",
            "dia concreto a predecir (e.g. dia 1 o el 2 o 3 o el 15 o el 30)DÍA CONCRETO: ",
            dcc.Input(id='dia-concreto-horasDiaConcreto', value='', type='text'),
            "Hora inicial formato(0 para las 12 noche, 1 para la una noche, 23 para las 11 noche)HORA INICIAL: ",
            dcc.Input(id='hora-inicial-horasDiaConcreto', value='', type='text'),
            "Hora final: formato(0 para las 12 noche, 1 para la una noche, 23 para las 11 noche) HORA FINAL:",
            dcc.Input(id='hora-final-horasDiaConcreto', value='', type='text'),
            "hasta que fecha predecir dia concreto formato (yyyy-mm-dd)FECHA FIN: ",
            dcc.Input(id='fecha-concreto-diaConcreto', value='', type='text'),
            html.Button(id='submit-horasDiaConcreto', type='submit', children='Predecir'),
            html.Br(), html.Br(),
            html.Div(id='output_div3')
        ]),
        html.Div([
            "Predicción de un número de días desde 2019-01-06 (e.g. introduzca fecha fin ) Fecha Fin: ",
            dcc.Input(id='hastafecha-todoslosdias', value='i', type='text'),
            html.Button(id='submit-todoslosdias', type='submit', children='Predecir'),
            html.Div(id='output_div4')
        ]),

    ], style={'padding': 10, 'flex': 4, 'color': colors['text']}),

    dcc.Graph(
        id='example-graph-2',
        figure=fig
    ),

])


# @app.callback(
#     Output('cities-radio', 'options'),
#     Input('options-radio', 'value'))
# def set_cities_options(selected_option):
#     return [{'label': i, 'value': i} for i in all_options[selected_option]]


@app.callback(
    Output('dd-output-container', 'children'),
    Input('dropdown-descricion', 'value')
)
def update_output(value):
    if value != 0 and value != '':
        return html.H4('id de tramo= ' + str(giveme_idtramo(value)))


def giveme_idtramo(value):
    if value != 0 and value != '' and trafico_union_semestres_sin2019.loc[
        trafico_union_semestres_sin2019['descripcion'] == value, 'idTramo'].unique().size > 0:
        idtramo = trafico_union_semestres_sin2019.loc[
        trafico_union_semestres_sin2019['descripcion'] == value, 'idTramo'].unique()[0]
        return idtramo


@app.callback(Output('output_div', 'children'),
              [Input('submit-diaConcreto', 'n_clicks')],
              [State('dia-diaConcreto', 'value')],
              [State('fecha-diaConcreto', 'value')],
              [State('IntenoOcupaoVeloci', 'value')],
              [State('dropdown-descricion', 'value')],
              )
def update_output2(clicks, diaConcreto_value, fechaDiaConcreto, quepredic, tramodes):
    if clicks is not None:
        print(clicks, diaConcreto_value, fechaDiaConcreto, quepredic)
        numeroDeDias = numOfDays('2019-01-06', fechaDiaConcreto)
        quepredic = quepredic
        idtramo = ''
        df = trafico_union_semestres_sin2019
        if tramodes != '':
            idtramo = giveme_idtramo(tramodes)
        if idtramo != '':
            print(idtramo)
            df = trafico_union_semestres_sin2019.loc[trafico_union_semestres_sin2019['idTramo'] == int(idtramo)]
            df = df.loc[df['fecha'].dt.day == int(diaConcreto_value)]
            if quepredic == "I" or quepredic == "i":
                df['media'] = df['intensidad']['mean']
            elif quepredic == "O" or quepredic == "o":
                df['media'] = df['ocupacion']['mean']
            elif quepredic == "V" or quepredic == "v":
                df['media'] = df['velocidad']['mean']

        df.set_index('fecha', inplace=True)
        ts = pd.DataFrame({'ds': df.index, 'y': df.media})
        prophet_plot(ts, numeroDeDias)
        # fig = go.Figure(data=[go.Scatter(x=ts['ds'], y=ts['y'])])
        # fig.update_layout(
        #     title=f"{quepredic} de {diaConcreto_value} de {fechaDiaConcreto}",
        #     xaxis_title="Fecha",
        #     yaxis_title=f"{quepredic}",
        #     font=dict(  family="Courier New, monospace", size=14, color="#7f7f7f"),
        #     paper_bgcolor=colors['background'],
        #     plot_bgcolor=colors['background'],
        #     xaxis=dict( gridcolor=colors['grid'],
        #                  tickfont=dict(color=colors['text'])),
        #     yaxis=dict( gridcolor=colors['grid'],
        #                  tickfont=dict(color=colors['text'])),
        #     legend_orientation="h",
        #     legend=dict(x=0, y=1.0, bgcolor=colors['background'],
        #                  bordercolor=colors['background'],
        #                  font=dict(color=colors['text'])))

def prophet_plot(ts, numeroDeDias):
    prophet = Prophet(changepoint_range=1)
    prophet.fit(ts)
    future = prophet.make_future_dataframe(periods=int(numeroDeDias))
    forecast = prophet.predict(future)
    fig = prophet.plot(forecast)
    fig.show()

@app.callback(Output('output_div2', 'children'),
              [Input('submit-horasConcretasTodos', 'n_clicks')],
              [State('hora-inicial-horasConcretasTodos', 'value')],
              [State('hora-final-horasConcretasTodos', 'value')],
              [State('fecha-horasConcretasTodos', 'value')],
              [State('IntenoOcupaoVeloci', 'value')],
              )
def update_output3(clicks, horaInicialHorasConcretasTodosDias, horaFinalHorasConcretasTodosDias,
                   fechaHorasConcretasTodosDias, intenoOcupaoVeloci):
    if clicks is not None:
        print(clicks, horaInicialHorasConcretasTodosDias, horaFinalHorasConcretasTodosDias,
              fechaHorasConcretasTodosDias, intenoOcupaoVeloci)


@app.callback(Output('output_div3', 'children'),
              [Input('submit-horasDiaConcreto', 'n_clicks')],
              [State('dia-concreto-horasDiaConcreto', 'value')],
              [State('hora-inicial-horasDiaConcreto', 'value')],
              [State('hora-final-horasDiaConcreto', 'value')],
              [State('fecha-concreto-diaConcreto', 'value')],
              [State('IntenoOcupaoVeloci', 'value')],
              )
def update_output4(clicks, diaConcreto, horaInicialHorasConcretasDiaConcreto, horaFinalHorasConcretasDiaConcreto,
                   fechaHorasConcretasDiaConcreto, intenoOcupaoVeloci):
    if clicks is not None:
        print(clicks, diaConcreto, horaInicialHorasConcretasDiaConcreto, horaFinalHorasConcretasDiaConcreto,
              fechaHorasConcretasDiaConcreto, intenoOcupaoVeloci)


@app.callback(Output('output_div4', 'children'),
              [Input('submit-todoslosdias', 'n_clicks')],
              [State('hastafecha-todoslosdias', 'value')],
              [State('IntenoOcupaoVeloci', 'value')],
              )
def update_output5(clicks, fechaHastaDia, intenoOcupaoVeloci):
    if clicks is not None:
        print(clicks, fechaHastaDia, intenoOcupaoVeloci)


def numOfDays(date1, date2):
    format = '%Y-%m-%d'
    d1 = datetime.strptime(date1, format).date()
    d2 = datetime.strptime(date2, format).date()
    print(d1)
    print(d2)
    return (d2 - d1).days


if __name__ == '__main__':
    app.run_server(debug=True)
