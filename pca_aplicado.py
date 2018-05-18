# -*- coding: utf-8 -*-
#
# 1.- Crea un fichero llamado pca_aplicado.py
#
# 2.- Carga los datos de digitos de sklearn (load_digits):
#               http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

digits = datasets.load_digits(n_class=10, return_X_y=False)
print(digits.target)
# 3.- Intena reducir el nivel de complicación de características usando el algoritmo de PCA

pca=PCA(n_components=16)
pca.fit(digits.data)

transformada = pca.transform(digits.data)

print(digits.data.shape)

print(transformada.data.shape)

# 4.- Aplica un algortimo de clasificación tras la transformación con los datos de etiquetas originales


# Prueba con Kmeans
print('\nAplicando PCA')

km=KMeans(n_clusters=10,max_iter=10000)

km.fit(transformada.data)

predicciones=km.predict(transformada.data)
#vemos las predicciones sobre el set de datos
print(predicciones)
score=metrics.adjusted_rand_score(digits.target,predicciones)
#Vemos el porcentaje de aciertos respecto a lo esperado
print("Score Modelo Kmeans:", score)

# Prueba con KNN

X_train, X_test, Y_train, Y_test=train_test_split(transformada, digits.target,
                                                 train_size=0.80, test_size=0.20, random_state=2)

knn=KNeighborsClassifier(n_neighbors=10,weights='distance')

#entrenamos al algoritmo con los datos (_train)
knn.fit(X_train,Y_train)

#comprobamos la validez del algortimo
score=knn.score(X_test,Y_test)
print ("Score Modelo KNN:",score)

lm= LinearRegression()

#Entrenamos el Modelo
lm.fit(X_train,Y_train)
score=lm.score(X_test,Y_test)
print("Score Modelo Regresion lineal:",score)

print('\nSin PCA, juego datos original')


km=KMeans(n_clusters=10,max_iter=10000)

km.fit(digits.data)

predicciones=km.predict(digits.data)
#vemos las predicciones sobre el set de datos
print(predicciones)
score=metrics.adjusted_rand_score(digits.target,predicciones)
#Vemos el porcentaje de aciertos respecto a lo esperado
print("Score Modelo Kmeans:", score)

X_train, X_test, Y_train, Y_test=train_test_split(digits.data, digits.target,
                                                 train_size=0.80, test_size=0.20, random_state=2)

knn=KNeighborsClassifier(n_neighbors=10,weights='distance')

#entrenamos al algoritmo con los datos (_train)
knn.fit(X_train,Y_train)

#comprobamos la validez del algortimo
score=knn.score(X_test,Y_test)
print ("Score Modelo KNN:",score)

lm= LinearRegression()


#Entrenamos el Modelo
lm.fit(X_train,Y_train)

score=lm.score(X_test,Y_test)
print("Score Modelo Regresion lineal:",score)