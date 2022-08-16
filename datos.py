import imghdr
from pydoc import render_doc
from re import template
from tkinter import Canvas
import pandas as pd
import matplotlib.pyplot as plot 
import numpy as np
from scipy import stats
#conversor de imagen
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Response
from matplotlib.figure import Figure
import seaborn as sns
#hasta aca va la de datos, ingresando a  flask
from flask import Flask, render_template, request, send_file
app=Flask(__name__)


#funciones del tratamiento de datos, toco ponerlas aca porque no las encontraba

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


def todelete(column,newlist):
    deleted = []
    for i in range(0,len(datacsv[column])):
        if (datacsv[column][i] < newlist[0]-1.5*newlist[2]):
            deleted.append(i)
        if (datacsv[column][i] > newlist[1] + 1.5*newlist[2]):
            deleted.append(i)
        
    return deleted
            
def deleting(column,todelete):
    datacsv2 = datacsv.drop(datacsv.index[todelete])
    return datacsv2

def plotframe(datacsvframe,column):
    
    plot.hist(datacsvframe[column])
    plot.subplots()
    plot.boxplot(datacsvframe[column])
    fig = plot.figure()
    ax =fig.add_subplot(111)
    res= stats.probplot(datacsvframe[column], dist=stats.norm,plot=ax)
    plot.subplots()
    
def framehist(datacsvframe,column):
    figura = plot.hist(datacsvframe[column])
    plot.show()
    return figura
    
def repeating(column):
    for i in range(0,len(column)-1):
        plotframe(deleting(column[i+1],todelete(column[i+1],quantiles(column[i+1]))),column[i+1])
        
##############################################################################
#hoja de rutas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    
    #este indica si esta activado o desactivado los atipicos
    atipicos = request.form['correcion']
    #activado-desactivado, perdo siempre debe estar seleccionado o dara error

    #recuperamos el factor alfa
    factoralfa = request.form['factoralfa']
    #devuelve numero entre 1 y 3, hay que corregirlo para que sea 1,5 a 3

    #variables de entrada, se recupera el primero seleccionado, se deberia recuperar la
    #entrada completa
    variableentrada = request.form.getlist('variable')
    #vi como recuperar la lista completa seleccionada

    #variable de salida, una sola
    variablesalida = request.form.get('variablesalida')
    #devuelve el texto

    #regresion, se hace igual que arriba
    regresion = request.form.get('regresion')
    #devuelve el texto

    #tamaño de entrenamiento, se mete un numero de 0 a 100
    tamañoentrenamiento = request.form['tamañoentrenamiento']
    #regresa el numero

    #modelo para una sola entrada de variable, y una sola salida
    if(len(variableentrada)==1 and variablesalida == "no"):
        if(atipicos=="activado"):
            fig=framehist(deleting(variableentrada[0],todelete(variableentrada[0],quantiles(variableentrada[0]))),variableentrada[0])
            ax=fig.subplots()
            ax.plot([1,2])
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            print(data)
            return render_template('index.html',data=data)
            
    return Response(img.getvalue(),mimetype='image/png')
    #render_template('index.html',data=tamañoentrenamiento)



if __name__=='__main__':
    app.run(debug=True)


file = 'abalone.csv'

data = pd.read_csv(file)
column = ['sex', 
          'length',
          'diameter',
          'height',
          'whole weight',
          'shucked weight',
          'viscera weight',
          'shell weight',
          'rings']
data.columns =column

def quantiles(column):
    newlist = []
    qr1 = (data[column].quantile(q=0.25))
    qr3 = (data[column].quantile(q=0.75))
    
    iqr = qr3-qr1
    newlist.append(qr1)
    newlist.append(qr3)
    newlist.append(iqr)
    return newlist


def todelete(column,newlist):
    deleted = []
    for i in range(0,len(data[column])):
        if (data[column][i] < newlist[0]-1.5*newlist[2]):
            deleted.append(i)
        if (data[column][i] > newlist[1] + 1.5*newlist[2]):
            deleted.append(i)
        
    return deleted
            
def deleting(column,todelete):

    data2 = data.drop(data.index[[todelete]])
    return data2

def plotframe(dataframe,column):
    
    plot.hist(dataframe[column])
    plot.subplots()
    plot.boxplot(dataframe[column])
    fig = plot.figure()
    ax =fig.add_subplot(111)
    res= stats.probplot(dataframe[column], dist=stats.norm,plot=ax)
    plot.subplots()
    

def repeating(column):
    for i in range(0,len(column)-1):
        plotframe(deleting(column[i+1],todelete(column[i+1],quantiles(column[i+1]))),column[i+1])
    
    
repeating(column)

