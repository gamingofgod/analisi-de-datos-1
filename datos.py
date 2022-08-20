from audioop import rms
import imghdr
from logging.handlers import DatagramHandler
from pydoc import render_doc
from re import template
from tkinter import Canvas
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib
matplotlib.use('Agg')
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
#decimos ubicacion del csv
file = "datos/abalone.csv"
rmse=''
#leemos el fichero
datacsv = pd.read_csv(file)

#establecemos las columnas
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

#esta funcion, recibe la columna y saca los qr1 qr3 y iqr
def quantiles(column):
    newlist = []
    qr1 = (datacsv[column].quantile(q=0.25))
    qr3 = (datacsv[column].quantile(q=0.75))

    iqr = qr3-qr1
    newlist.append(qr1)
    newlist.append(qr3)
    newlist.append(iqr)
    return newlist

#genera la lista de indices que se deben eliminar de una columna, recibe la lista anteriorir de qr's
def todelete(column, newlist, factoralfa):
    deleted = []
    for i in range(0, len(datacsv[column])):
        if (datacsv[column][i] < newlist[0]-int(factoralfa)*newlist[2]):
            deleted.append(i)
        if (datacsv[column][i] > newlist[1] + int(factoralfa)*newlist[2]):
            deleted.append(i)
    return deleted

#necesito para generar el modelo con el mismo len en cada columna
#entonces lo que hago es obtener la lista de todos los atipicos, sumados todas las listas
#idividuales de cada columna
#recibe el factor alfa (1.5 a 3) y la lista de columnas
def todeletemodel(listaColumnas, factoralfa):
    deletedtotal=[]
    listasumada=[]
    for i in range(len(listaColumnas)):
        newlist=quantiles(listaColumnas[i])
        listasumada = todelete(listaColumnas[i], newlist, factoralfa) + listasumada
    deletedtotal = set(listasumada)
    deletedtotal=list(deletedtotal)
    return deletedtotal

#elimina los indices, recibe la columna y la lista a eliminar
def deleting(column, todelete):
    datacsv2 = datacsv.drop(datacsv.index[todelete])
    return datacsv2

##este metodo se encarga de mostrar el antes y despues de el historigrama
#recibe el dataframe modificado (limpio), la columna a tratar y el dataframe original
def framehist(datacsvframe, column, dataframeoriginal):
    fig = plot.figure()
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

##este metodo se encarga de generar el antes y despues de el diagrama de bigotes
def frameboxplot(datacsvframe, column, dataframeoriginal):
    fig = plot.figure()
    figura = plot.boxplot(datacsvframe[column])
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)
    plot.subplots()
    figura = plot.boxplot(dataframeoriginal[column])
    ubicacion2 = 'datos/static/grafica2.png'
    plot.savefig(ubicacion2)

##este metodo se encarga de generar el antes y despues de el diagrama de normalizacion
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

#funcion corazon para cuando la persona no quiera ver atipicos
def corazon():
    plot.subplots()
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
def framescatter(datacsvframeA,datacsvframeB, column,column2, dataframeoriginal):
    fig = plot.figure()
    plot.scatter(list(datacsvframeA),list(datacsvframeB))
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)
    plot.subplots()
    plot.scatter(dataframeoriginal[column], dataframeoriginal[column2])
    ubicacion2 = 'datos/static/grafica2.png'
    plot.savefig(ubicacion2)

#pal modelo
def framescattermodel(modelogeneradolimpio, modelogeneradosucio, variablelimpia, variablesucia):
    fig = plot.figure()
    plot.scatter(modelogeneradolimpio, variablelimpia)
    ubicacion = 'datos/static/grafica.png'
    plot.savefig(ubicacion)
    plot.subplots()
    plot.scatter(modelogeneradosucio, variablesucia)
    ubicacion2 = 'datos/static/grafica2.png'
    plot.savefig(ubicacion2)

#funcion para que el alfa se asigne correctamente
def corregirfactoralfa(factoralfa):
    if(factoralfa=="0"):
        return 1.5
    if(factoralfa=="1"):
        return 2
    if(factoralfa=="2"):
        return 2.5
    if(factoralfa=="3"):
        return 3

#aca lo que aho es que como el modelo tiene menos cantidad de elementos, para poder
#producir la imagen necesitamos capar la cantidad elementos en el modelo limpio
def construirnuevalimpia(modelo,alimpiar):
    lenmodelo=len(modelo)
    alimpiar=alimpiar[0:lenmodelo]
    return alimpiar

#funciones para el modelo
#modelo de una entrada y una salida, con regresion lineal (porque no hemos visto mas)
def modelounoauno(tamañoentrenamiento,X,Y):
    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        Y,
                                        train_size   = corregirentrenamiento(tamañoentrenamiento),
                                    )
    modelo = LinearRegression()
    modelo.fit(X = np.array(X_train).reshape(-1, 1), y = y_train)
    datosobtenidos = modelo.predict(X = np.array(X_test).reshape(-1,1))
    rmse = mean_squared_error(y_true  = y_test, y_pred  = datosobtenidos)
    return datosobtenidos
    
#modelo uno a muchos
#x es una lista
def modelounoamuchos(tamañoentrenamiento,X,Y):
    X = X.drop(['sex'], axis=1)
    Y = Y[0:len(X)]
    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        Y,
                                        train_size   = corregirentrenamiento(tamañoentrenamiento),
                                    )
    modelo = LinearRegression()
    modelo.fit(X = np.array(X_train), y = y_train)
    datosobtenidos = modelo.predict(X = np.array(X_test))
    rmse = mean_squared_error(y_test,datosobtenidos)
    return datosobtenidos

#funcion de correcion de tamañode entranamiento
def corregirentrenamiento(tamañoentrenamiento):
    return tamañoentrenamiento/100
    
##############################################################################
# hoja de rutas de flask

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

    #esto recupera que quiere la persona, modelo o scatter directamente
    modelo = request.form['modelo2']
    #retorna si, no

    # regresion, se hace igual que arriba
    regresion = request.form.get('regresion')
    # devuelve el texto

    # tamaño de entrenamiento, se mete un numero de 0 a 100
    tamañoentrenamiento = request.form['tamañoentrenamiento']
    # regresa el numero

    #seleccionar grafica
    tipografica = request.form.get('tipografica')

    # este if determina si solo hay una variable de entrada y no hay variables de salida
    #en caso de solo haberla, genera ambas imagenes, y solo en caso
    #de tener desactivado los atipicos sobreescribiria la segunda iamgen por un corazon
    if (len(variableentrada) == 1 and variablesalida == "no"):
        if(tipografica=="dispersion"):
            fig = framehist(deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa))), variableentrada[0], datacsv)
        if(tipografica=="diagrama de bigotes"): 
            fig = frameboxplot(deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa))), variableentrada[0], datacsv)
        if(tipografica=="normalizacion"):
            fig = frameprobplot(deleting(variableentrada[0], todelete(variableentrada[0], quantiles(variableentrada[0]), corregirfactoralfa(factoralfa))), variableentrada[0], datacsv)
        if (atipicos == "desactivado"):
            corazon()

    #este if revisa si hay una variable de entrada y una de salida, se genera la imagen de relacion y el modelo
    if (len(variableentrada) == 1 and variablesalida != "no"):
        if(modelo=="si"):
            if (atipicos == "activado"):
                #este bloque, esta raro, creo que a eso se debe la imagen, hay que revisarlo
                muchasJuntas=variableentrada
                muchasJuntas.append(variablesalida)
                modelogeneradolimpio = modelounoauno(int(tamañoentrenamiento),deleting(variableentrada[0], todeletemodel(muchasJuntas, corregirfactoralfa(factoralfa)))[variableentrada[0]],deleting(variablesalida, todeletemodel(muchasJuntas, corregirfactoralfa(factoralfa)))[variablesalida])
                modelogeneradosucio = modelounoauno(int(tamañoentrenamiento),datacsv[variableentrada[0]],datacsv[variablesalida])
                var = deleting(variablesalida, todeletemodel(muchasJuntas, corregirfactoralfa(factoralfa)))[variablesalida]
                meter=construirnuevalimpia(modelogeneradolimpio,var)
                var2 = datacsv[variablesalida]
                meter2=construirnuevalimpia(modelogeneradosucio,var2)
                framescattermodel(list(modelogeneradolimpio),modelogeneradosucio,list(meter),meter2)
            else:
                corazon()
        if(modelo=="no"):
            muchasJuntas=variableentrada
            muchasJuntas.append(variablesalida)
            framescatter(deleting(variableentrada[0],todeletemodel(muchasJuntas, corregirfactoralfa(factoralfa)))[variableentrada[0]],deleting(variablesalida, todeletemodel(muchasJuntas, corregirfactoralfa(factoralfa)))[variablesalida],variableentrada[0],variablesalida,datacsv)
            if (atipicos == "desactivado"):
                corazon()
    else:
        if (atipicos == "activado" and variablesalida !="no"):
            #x es datafream
            #y es una columna
            variableentrada.append(variablesalida)
            filasaeliminar=todeletemodel(variableentrada,factoralfa)
            resultadoDF=deleting("ola",filasaeliminar)
            modelogeneradolimpio =modelounoamuchos(corregirentrenamiento(int(tamañoentrenamiento)),resultadoDF,deleting("ola",todelete(variablesalida,quantiles(variablesalida),corregirfactoralfa(factoralfa)))[variablesalida])
            modelogeneradosucio =modelounoamuchos(corregirentrenamiento(int(tamañoentrenamiento)),datacsv,datacsv[variablesalida])
            variablelimpia=deleting("ola",todelete(variablesalida,quantiles(variablesalida),corregirfactoralfa(factoralfa)))[variablesalida]
            if(len(modelogeneradolimpio)>len(variablelimpia)):
                modelogeneradolimpio = modelogeneradolimpio[0:len(variablelimpia)]
            else:
                variablelimpia = variablelimpia[0:len(modelogeneradolimpio)]
            listakk = list(datacsv[variablesalida])
            if(len(datacsv[variablesalida])>len(modelogeneradosucio)):
                listakk = list(datacsv[variablesalida])
                listakk=listakk[0:len(modelogeneradosucio)]
            else:
                modelogeneradosucio = modelogeneradosucio[0:len(datacsv[variablesalida])]
            framescattermodel(modelogeneradolimpio,modelogeneradosucio,variablelimpia,listakk)
        else:
            corazon()
    #con este return, solo le digo que renderice con las 2 imagenes cuales quiera
    #ya depende de cada metodo de los de arriba de las rutas de flask guardar las imagenes que son
    return render_template('index.html', datas=["../static/grafica2.png", "../static/grafica.png",rmse])


if __name__ == '__main__':
    app.run(debug=True)
