# 1.- Crea un nuevo fichero sk.data.py
#
# 2.- Carga las dependencias de sklearn datasets

import sklearn
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets
from pandas import scatter_matrix
import pandas as pd
import numpy as np

# 3.- Carga los datos de Iris

iris = datasets.load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(iris.data)
print(data1.describe)
print(data1.shape)

# 4.- Crea las gráficas de cajas, histograma y scatter de los datos de iris

fig= plt.figure()
plt.boxplot(iris.data, labels=iris.feature_names)
plt.show()
plt.hist(iris.data)
plt.show()
data1.plot.scatter(x='petal width (cm)', y='petal length (cm)')
plt.show()
scatter_matrix(data1)
plt.show()

# 5.- Haz lo mismo con los datos de boston

boston = datasets.load_boston()
columnas = []
for fe in boston.feature_names:
    columnas = columnas + [fe]

print(boston.feature_names)
print(columnas)
data1 = pd.DataFrame(data= np.c_[boston['data'], boston['target']],
                     columns= columnas + ['target'])

fig= plt.figure()
plt.boxplot(boston.data, labels=boston.feature_names)
plt.show()
plt.hist(boston.data)
plt.show()
scatter_matrix(data1)
plt.show()

print(data1["CRIM"])
# 6.- Si te ves con ganas haz las gráfica con Bokeh

from bokeh.plotting import figure,output_file,show
from bokeh.models import ColumnDataSource

#Genera los datos
data = {'x_values': [1, 2, 3, 4, 5],
        'y_values': [6, 7, 2, 3, 6]}

#Crea la fuente de datos
source = ColumnDataSource(data=data)

#Crea la figura
p = figure()
#Crea un círculo
p.circle(x='x_values', y='y_values', source=source)
#Muestra la gráfica
show(p)

import bokeh
from bokeh.plotting import figure,output_file,show
print('bokeh: {}'.format(bokeh.__version__))

p = figure(title="Boston")

BoxPlot(data1['CRIM'], label='CRYM',
            title="MPG Summary (grouped by CYL)")




