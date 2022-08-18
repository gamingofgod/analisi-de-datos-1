import imghdr
from pydoc import render_doc
from re import template
from tkinter import Canvas
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from scipy import stats
# conversor de imagen
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Response
from matplotlib.figure import Figure
import seaborn as sns
# hasta aca va la de datos, ingresando a  flask
from flask import Flask, render_template, request, send_file
app = Flask(__name__)

######  paquetes de analitica de datos
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# funciones del tratamiento de datos, toco ponerlas aca porque no las encontraba

file = "datos/abalone.csv"

datacsv = pd.read_csv(file)


column = ['sex',
          'length',
          'diameter',
          'height',
          'whole weight',
          'shucked weight',
          'viscera weight',
          'shell weight',
          'rings']
datacsv.columns = column


def quantiles(column):
    newlist = []
    qr1 = (datacsv[column].quantile(q=0.25))
    qr3 = (datacsv[column].quantile(q=0.75))

    iqr = qr3-qr1
    newlist.append(qr1)
    newlist.append(qr3)
    newlist.append(iqr)
    return newlist


def todelete(column, newlist, factoralfa):
    deleted = []
    for i in range(0, len(datacsv[column])):
        if (datacsv[column][i] < newlist[0]-factoralfa*newlist[2]):
            deleted.append(i)
        if (datacsv[column][i] > newlist[1] + factoralfa*newlist[2]):
            deleted.append(i)

    return deleted


def deleting(column, todelete):
    datacsv2 = datacsv.drop(datacsv.index[todelete])
    return datacsv2

##este metodo se encarga de mostrar el antes y despues de el historigrama
#recibe el dataframe modificado (limpio), la columna a tratar y el dataframe original
def framehist(datacsvframe, column, dataframeoriginal):
    figura = plot.hist(datacsvframe[column])
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)
    plot.subplots()
    figura = plot.hist(dataframeoriginal[column])
    ubicacion2 = 'datos/static/grafica2.png'
    plot.savefig(ubicacion2)

#retorna 2 ubicaciones genericas, de hecho, todo retorna dos ubicaciones que se renderizan
#en el html, la diferencia es que cada metodo incluye dibujar la grafica y guardarla
#de acuerdo al metodo se guardaran las 2 imagenes y ya

def frameboxplot(datacsvframe, column, dataframeoriginal):
    figura = plot.boxplot(datacsvframe[column])
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)
    plot.subplots()
    figura = plot.boxplot(dataframeoriginal[column])
    ubicacion2 = 'datos/static/grafica2.png'
    plot.savefig(ubicacion2)

#ya funciona

def frameprobplot(datacsvframe, column, dataframeoriginal):
    fig = plot.figure()
    ax = fig.add_subplot(111)
    res = stats.probplot(datacsvframe[column], dist=stats.norm, plot=ax)
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)
    plot.subplots()
    fig = plot.figure()
    ax = fig.add_subplot(111)
    res = stats.probplot(dataframeoriginal[column], dist=stats.norm, plot=ax)
    ubicacion2 = 'datos/static/grafica2.png'
    plot.savefig(ubicacion2)

#funcion corazon para cuando la persona no quiera ver sin atipicos
def corazon():
    x = np.linspace(-1,1,1000)
    y1 = np.sqrt(x * x) + np.sqrt(1 - x * x)
    y2 = np.sqrt(x * x) - np.sqrt(1 - x * x)
    plot.plot(x, y1, c='r', lw = 2)
    plot.plot(x, y2, c='r', lw = 2)
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)

#funciones de comparacion de 2 o mas variables, toca ahcer una de 1 variable de entrada
#y una de salida, y otro metoodo para n variables de entrada y una de salida
#el codigo cambia ligeramente

#por ahora solo la imagen

def framescatter(datacsvframe, column,column2, dataframeoriginal):
    plot.scatter(datacsvframe[column], datacsvframe[column2])
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)
    plot.subplots()
    plot.scatter(dataframeoriginal[column], dataframeoriginal[column2])
    ubicacion2 = 'datos/static/grafica2.png'
    plot.savefig(ubicacion2)

#funciones para el modelo
#modelo de una entrada y una salida
def modelounoauno(variableentrada,variablesalida,tipoderegresion,tamañoentrenamiento,factoralfa):

    X=deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa)))[variableentrada]
    Y=deleting(variablesalida, todelete(variablesalida, quantiles(variablesalida), corregirfactoralfa(factoralfa)))[variablesalida]

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        Y,
                                        train_size   = 0.5,
                                    )
    modelo = LinearRegression()
    modelo.fit(X = np.array(X_train).reshape(-1, 1), y = y_train)
            ##se debe retornar
    datosobtenidos = modelo.predict(X = np.array(X_test).reshape(-1,1))
    print(datosobtenidos)
    rmse = mean_squared_error(y_true  = y_test, y_pred  = predicciones)
    print(rmse)
            #############
    if(tipoderegresion == "polinomica"):
        print("hola")
    

#funcion para que el alfa se asigne correctamente

def corregirfactoralfa(factoralfa):
    if(factoralfa=="0"):
        return 1.5
        print("entro")
    if(factoralfa=="1"):
        return 2
    if(factoralfa=="2"):
        return 2.5
    if(factoralfa=="3"):
        return 3
#funcion de correcion de tamañode entranamiento

def corregirentrenamiento(tamañoentrenamiento):
    return tamañoentrenamiento/100
    
##############################################################################
# hoja de rutas


@app.route('/')
def index():
    return render_template('index.html', datas="")


@app.route('/visualize', methods=['POST'])
def visualize():

    # este indica si esta activado o desactivado los atipicos
    atipicos = request.form['correcion']
    # activado-desactivado, perdo siempre debe estar seleccionado o dara error

    # recuperamos el factor alfa
    factoralfa = request.form['factoralfa']
    # devuelve numero entre 1 y 3, hay que corregirlo para que sea 1,5 a 3

    # variables de entrada, se recupera el primero seleccionado, se deberia recuperar la
    # entrada completa
    variableentrada = request.form.getlist('variable')
    # vi como recuperar la lista completa seleccionada

    # variable de salida, una sola
    variablesalida = request.form.get('variablesalida')
    # devuelve el texto

    # regresion, se hace igual que arriba
    regresion = request.form.get('regresion')
    # devuelve el texto

    # tamaño de entrenamiento, se mete un numero de 0 a 100
    tamañoentrenamiento = request.form['tamañoentrenamiento']
    # regresa el numero

    #seleccionar grafica
    tipografica = request.form.get('tipografica')

    # modelo para una sola entrada de variable, y una sola salida
    if (len(variableentrada) == 1 and variablesalida == "no"):
        if (atipicos == "activado"):
            if(tipografica=="dispersion"):
                fig = framehist(deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa))), variableentrada[0], datacsv)
            if(tipografica=="diagrama de bigotes"): 
                fig = frameboxplot(deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa))), variableentrada[0], datacsv)
            if(tipografica=="normalizacion"):
                fig = frameprobplot(deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa))), variableentrada[0], datacsv)
        else:    
            corazon()
    if (len(variableentrada) == 1 and variablesalida != "no"):
        if (atipicos == "activado"):
            fig = framescatter(deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa))),variableentrada[0],variablesalida,datacsv)
        else:
            corazon()
    modelounoauno(variableentrada,variablesalida,regresion,tamañoentrenamiento,corregirfactoralfa(factoralfa))
    #con este return, solo le digo que renderice con las 2 imagenes cuales quiera
    #ya depende de cada metodo de los de arriba de las rutas de flask guardar las imagenes que son
    return render_template('index.html', datas=["../static/grafica2.png", "../static/grafica.png"])


if __name__ == '__main__':
    app.run(debug=True)
