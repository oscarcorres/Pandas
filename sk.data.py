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
plt.boxplot(iris.data)
plt.show()
plt.hist(iris.data)
plt.show()
# iris.plot.scatter(x='petal_width (cm)', y='petal_length (cm)')
scatter_matrix(data1)
plt.show()

# 5.- Haz lo mismo con los datos de boston
#
# 6.- Si te ves con ganas haz las gráfica con Bokeh