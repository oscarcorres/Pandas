# 1.- Crea un fichero llamado clasificacion_knn.py
#
# 2.- Coge los datoa de irirs y cargalos

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()

# 3.- Utiliza el algortimo de KNN para obtenerun modelo y calcula su nivel de acierto

X_train, X_test, Y_train, Y_test=train_test_split(iris['data'],iris['target'],
                                                  train_size=0.80, test_size=0.20)

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,Y_train)

score=knn.score(X_test,Y_test)
print (score)

# 4.- Busca mediante un bucle el número de vecinos cercanos que mejor se ajusta al modelo, intenta evitar pa puntuacion 1.0 porque estaríamos en un caso de sobreajuste

test_modelos= {}
mejor = (0,0)
for x in range(2,15):
    knn=KNeighborsClassifier(n_neighbors=x)
    knn.fit(X_train, Y_train)
    score = knn.score(X_test, Y_test)
    test_modelos[x] = score
    if mejor[1] < score and score<1:
        mejor = (x,score)
print(test_modelos)
print('El mejor numero de vecinos es', mejor[0], ' con un score de ', mejor[1])
