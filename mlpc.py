# -*- coding: utf-8 -*-
#
# Crea un fichero llamado mlpc.py
# Carga los datos de cancer de sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

cancer=datasets.load_breast_cancer()
X=cancer.data
Y=cancer.target

# Crea una red neuronal MLPC

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,random_state=2)

red=MLPClassifier(max_iter=1500, hidden_layer_sizes=(5,7), activation="relu", solver="lbfgs", shuffle=False, random_state=2)

ret=red.fit(X_train,Y_train)
print(ret)
score=red.score(X_test, Y_test)

print("Score:",score)

# Intenta obtener el mejor resultado posible de predicción

mejor = 0
for activacion in ('identity', 'logistic', 'tanh', 'relu'):
    print ('Activacion',activacion)
    for solver in ("adam","lbfgs",'sgd'):
        for capa in range(2,10):
            for neuronas in range(2,15):
                red=MLPClassifier(max_iter=1500, hidden_layer_sizes=(capa,neuronas),
                                  activation=activacion, solver=solver, shuffle=False, random_state=2)
                ret=red.fit(X_train,Y_train)
                score=red.score(X_test, Y_test)
                print("Con",capa,"capas y ",neuronas,"neuronas - resultado:",score)
                if score > mejor:
                    mejor = score
                    mejorcon = (capa,neuronas,solver,activacion)

print("Score:",mejor,"con",mejorcon[0],"capas y",mejorcon[1],"neuronas, solver", mejorcon[2], 'y activacion', mejorcon[3])