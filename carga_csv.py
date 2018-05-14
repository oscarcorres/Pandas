# 1.- Crea un nuevo fichero python llamada carga_csv.py

# 2.- Importa las bibliotecas de pandas

import pandas as pd

# 3.- Carga el fichero csv desde python

iris = pd.read_csv('./csv/iris.data.csv', header=None)

# 4.- Imprime la forma y las etiquetas del conjunto de datos : iris.data.csv

print('\nForma del conjunto de datos: ',iris.shape)

print(iris.axes)

tag = iris.groupby(4).count()

print(tag)
