import pandas as pd
import matplotlib.pyplot as plot 
import numpy as np
from scipy import stats


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
