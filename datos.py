import pandas as pd
import matplotlib.pyplot as plot
from scipy import stats

archivo="abalone.csv"

datos=pd.read_csv(archivo)

columnas=["sex","lenght","diameter","heigth","whole weight","shucked weight","viscera weight","shell weigth","rings"]

##leemos e importamos

datos.columns=columnas

##ponemos los cabezales

plot.hist(datos["lenght"])
plot.subplots()
plot.boxplot(datos["rings"])

##aca hacemos dos distribuciones y separamos con la linea 18

fig=plot.figure()
ax=fig.add_subplot(111)
res=stats.probplot(datos["lenght"],dist=stats.norm,plot=ax)
##aca lo que hicimos fue establecer una figura en blanco
##despues con ese 311 se ubian diferente cantidad de figuras
##hace la distribucion normal de la linea de datos dada
##se incluye el tipo de distribucion y lo obtenido se sube a ax

##identificacion de cuartiles
