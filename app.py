# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output
from fbprophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import pandas as pd
import self as self
import seaborn as sns
sns.set(color_codes=True)

app = Dash(__name__)

trafico_union_semestres = pd.read_pickle('trafico_union_semestres.plk')
trafico_union_semestres.fecha = pd.to_datetime(trafico_union_semestres.fecha)
descripcion = trafico_union_semestres['descripcion'].unique()
idTramoDescripcion = trafico_union_semestres[['descripcion','idTramo']].drop_duplicates()
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
        children='Prediccion de Tráfico',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children=[
        html.Label('Seleccione una descripción de tramo para predecir:'),
        dcc.Dropdown(descripcion, 'Pintor Sorolla, Nª 1', id='dropdown-descricion'),
        html.Div(id='dd-output-container'),

        html.Br(),
        html.Label('Seleccione opcións de visualización:'),
        dcc.RadioItems(['Prediccion del dia entero concreto (e.g. día 25 o día 14 de todos los meses)',
                        'Predicción de alguna/s horas concretas de todos los días (e.g. de hora inicial 18 a final 19)',
                        'Predicción de todos los dias de un número de días (e.g los próximos 30 o 365 días)',
                        'algúna/s horas concretas de un día en concreto (de 7 a 8 todos los días 15)'
                        ],"", id='options-radio'),
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
     return idTramoDescripcion.loc[idTramoDescripcion['descripcion'] == {value}, 'idTramo'].unique()

if __name__ == '__main__':
    app.run_server(debug=True)