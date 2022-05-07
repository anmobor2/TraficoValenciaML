import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import self as self

sns.set(color_codes=True)
import pickle
import fbprophet
from fbprophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

class traffic_prediction:
    def run(self):
        trafico_union_semestres = pd.read_pickle('trafico_union_semestres.plk')
#        trafico_union_semestres.head()
        trafico_union_semestres.fecha = pd.to_datetime(trafico_union_semestres.fecha)
        trafico_union_semestres.dtypes
        trafico_union_semestres.fecha.dt.year.value_counts()
#        trafico_union_semestres.head()

        print("Por favor introduzca tramo: ")
        tramo = input("Por favor introduzca tramo: ")
        dias = "";
        hora = "";
        horaInicial = -1;
        horaFinal = -1;
        numeroDeDias = -1;
        dia = "";
        todos = False

        diaUHora = input(
            "Prediccion del dia entero concreto D/d o de alguna/s horas concretas de todos los días H/h o todos los dias todas las horas T/t o algúna/s horas de un día en concreto DH/dh")
        if diaUHora == "D" or diaUHora == "d":
            dia = input("Por favor introduce el dia en formato dd (ejemplo 25)")
        elif diaUHora == "H" or diaUHora == "h":
            horaInicial = input("Por favor introduzca hora inicial tramo: 0-23 ")
            horaFinal = input("Por favor introduzca hora inicial final: 0-23 ")
        elif diaUHora == "T" or diaUHora == "t":
            todos = "Todos los dias"
        elif diaUHora == "DH" or diaUHora == "dh" or diaUHora == "dH" or diaUHora == "Dh":
            dia = input("Por favor introduce el dia en formato dd (ejemplo 25)")
            horaInicial = input("Por favor introduzca hora inicial tramo: 0-23 ")
            horaFinal = input("Por favor introduzca hora inicial tramo: 0-23 ")

        numeroDeDias = input("Número de días a predecir")

        quepredic = input("Que desea predecir: Intensidad(I), Ocupación(O), o Velocidad(V)")

        print("tramo:", tramo, "Hora inicial: ", horaInicial, "Hora final:", horaFinal, "dias ", dias,
              "Dias a predecir", todos, "Número de dia", numeroDeDias)


        print("tramo:", tramo)
        df = trafico_union_semestres.loc[trafico_union_semestres['idTramo'] == int(tramo)]
        calle = df.descripcion.unique()[0]
        print("Calle: " + calle + "\n")
        # calleCorrecta = input("¿Desea predecir el tramo de la calle "+calle+"? S/N")

        if diaUHora == "H" or diaUHora == "h":
            df = df.loc[df['hora'].between(int(horaInicial), int(horaFinal))]
        elif diaUHora == "D" or diaUHora == "d":
            df = df.loc[df['fecha'].dt.day == int(dia)]
        elif diaUHora == "T" or diaUHora == "t":
            df = df.loc[df['fecha'].dt.day == int(dia)]
            df = df.loc[df['hora'].between(int(horaInicial), int(horaFinal))]
        elif diaUHora == "T" or diaUHora == "t":
            df = df

        if quepredic == "I" or quepredic == "i":
            df['media'] = df['intensidad']['mean']
        elif quepredic == "O" or quepredic == "o":
            df['media'] = df['ocupacion']['mean']
        elif quepredic == "V" or quepredic == "v":
            df['media'] = df['velocidad']['mean']

  #      df.head()

        df.fecha.dt.year.value_counts()

        df.set_index('fecha', inplace=True)

        ts = pd.DataFrame({'ds': df.index, 'y': df.media})

  #      ts.tail()

        prophet = Prophet(changepoint_range=1, weekly_seasonality = False)
        prophet.fit(ts)

        future = prophet.make_future_dataframe(periods=int(numeroDeDias))
        forecast = prophet.predict(future)

        fig = prophet.plot(forecast)

        fig2 = prophet.plot_components(forecast)

        plot_plotly(prophet, forecast)



if __name__ == '__main__':
    traffic_prediction.run(self)