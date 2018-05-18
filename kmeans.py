# -*- coding: utf-8 -*-
#
# 1.- Crea un fichero llamado kmeans.py
#
# 2.- Carga los datos de cancer de pecho en un dataframe

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pnd

cancer=datasets.load_breast_cancer()
X=cancer.data
Y=cancer.target

print(X, cancer.keys())

# 3.- Intenta aplicar el algoritmo Kmeans y mira a ver si cumple con los requisitos de clasificación
#     con los datos de etiquetas originales

km=KMeans(n_clusters=2,max_iter=10000)

km.fit(X)

predicciones=km.predict(X)
#vemos las predicciones sobre el set de datos
print(predicciones)
score=metrics.adjusted_rand_score(Y,predicciones)
#Vemos el porcentaje de aciertos respecto a lo esperado
print(score)

# 4.- Grafica los resultados en un diagrama que marque la clasificación


sns.set(style="ticks")
columnas = [x for x in cancer.feature_names] + ['target']
df = pnd.DataFrame(data=np.c_[cancer.data, cancer.target], columns=columnas)
df.target=km.predict(X)
plot=sns.pairplot(df, hue="target")

plt.show()