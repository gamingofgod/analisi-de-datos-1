import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats

archivo="abalone.csv"

datos=pd.read_csv(archivo)

columnas=["sex","lenght","diameter","heigth","whole weight","shucked weight","viscera weight","shell weigth","rings"]

##leemos e importamos

datos.columns=columnas



##ponemos los cabezales

#plot.hist(datos["lenght"])
#plot.subplots()
#plot.boxplot(datos["rings"])

##aca hacemos dos distribuciones y separamos con la linea 18

#fig=plot.figure()
#ax=fig.add_subplot(111)
#res=stats.probplot(datos["lenght"],dist=stats.norm,plot=ax)

##aca lo que hicimos fue establecer una figura en blanco
##despues con ese 311 se ubian diferente cantidad de figuras
##hace la distribucion normal de la linea de datos dada
##se incluye el tipo de distribucion y lo obtenido se sube a ax

##identificacion de cuartiles

df=pd.DataFrame(datos)


def quartiles(columna):
    lista=[]
    q75=np.percentile(df[columna], [75])
    q25=np.percentile(df[columna], [25])
    lista.append(q25)
    lista.append(q75)
    iqr=q75-q25
    lista.append(iqr)
    return lista

def atipicos(columna,lista):
    listaeliminados=[]
    #tiene los id de las columnas a eliminar
    for i in range(0,len(df[columna])):
        if (df[columna][i]<lista[0]-1*lista[2]):
            listaeliminados.append(i)
        if (df[columna][i]>lista[1]+1*lista[2]):
            listaeliminados.append(i)
    return listaeliminados

def eliminacion(listaatipicos,datos):
    datanuevo=datos.drop(datos.index[[listaatipicos]])
    return datanuevo


def grafica (dataframetotal):
    
    for i in range(len(columnas)-1):
        plot.hist(datos[columnas[i+1]])
        plot.subplots()
        plot.boxplot(datos[columnas[i+1]])
       
        fig=plot.figure()
        ax=fig.add_subplot(111)
        res=stats.probplot(datos[columnas[i+1]],dist=stats.norm,plot=ax)
        plot.subplots()

#lenght

dfnuevo = eliminacion(atipicos("lenght",quartiles("lenght")), datos)


plot.hist(dfnuevo[columnas[1]])

plot.subplots()
plot.boxplot(dfnuevo[columnas[1]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo[columnas[1]],dist=stats.norm,plot=ax)
plot.subplots()

#diameter

dfnuevo2 = eliminacion(atipicos("diameter",quartiles("diameter")), datos)


plot.hist(dfnuevo2[columnas[2]])

plot.subplots()
plot.boxplot(dfnuevo2[columnas[2]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo2[columnas[2]],dist=stats.norm,plot=ax)
plot.subplots()


#heigth

dfnuevo3 = eliminacion(atipicos("heigth",quartiles("heigth")), datos)


plot.hist(dfnuevo3[columnas[3]])

plot.subplots()
plot.boxplot(dfnuevo3[columnas[3]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo3[columnas[3]],dist=stats.norm,plot=ax)
plot.subplots()

#whole weight

dfnuevo4 = eliminacion(atipicos("whole weight",quartiles("whole weight")), datos)


plot.hist(dfnuevo4[columnas[4]])

plot.subplots()
plot.boxplot(dfnuevo4[columnas[4]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo4[columnas[4]],dist=stats.norm,plot=ax)
plot.subplots()

#shucked weight

dfnuevo5 = eliminacion(atipicos("shucked weight",quartiles("shucked weight")), datos)


plot.hist(dfnuevo5[columnas[5]])

plot.subplots()
plot.boxplot(dfnuevo5[columnas[5]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo5[columnas[5]],dist=stats.norm,plot=ax)
plot.subplots()

#viscera weight

dfnuevo6 = eliminacion(atipicos("viscera weight",quartiles("viscera weight")), datos)


plot.hist(dfnuevo6[columnas[6]])

plot.subplots()
plot.boxplot(dfnuevo6[columnas[6]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo6[columnas[6]],dist=stats.norm,plot=ax)
plot.subplots()

#shell weigth

dfnuevo7 = eliminacion(atipicos("shell weigth",quartiles("shell weigth")), datos)


plot.hist(dfnuevo7[columnas[7]])

plot.subplots()
plot.boxplot(dfnuevo7[columnas[7]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo7[columnas[7]],dist=stats.norm,plot=ax)
plot.subplots()

#rings

dfnuevo8 = eliminacion(atipicos("rings",quartiles("rings")), datos)


plot.hist(dfnuevo8[columnas[8]])

plot.subplots()
plot.boxplot(dfnuevo8[columnas[8]])

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(dfnuevo8[columnas[8]],dist=stats.norm,plot=ax)
plot.subplots()














#print(len(eliminacion(atipicos("lenght",quartiles("lenght")), df)))      

#nuevodf=eliminacion(atipicos("lenght",quartiles("lenght")), df)

#print(len(eliminacion(atipicos("diameter",quartiles("diameter")), nuevodf)))


#print(atipicos("diameter", quartiles("diameter")))
    
    

##identificacion de iqr
# iqr=q75-q25

##eliminacion de datos atipicos
