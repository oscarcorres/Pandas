# -*- coding: utf-8 -*-
#
# Crea un fichero llamado mlpc.py
# Carga los datos de cancer de sklearn

from sklearn import datasets
import numpy as np
import tflearn

cancer=datasets.load_breast_cancer()
X=cancer.data
Y=cancer.target

# Crea una red neuronal MLPC

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(X, Y, n_epoch=50, batch_size=16, show_metric=True)

# Intenta obtener el mejor resultado posible de predicción
"""
mejor = 0
for activacion in ('identity', 'logistic', 'tanh', 'relu'):
    print ('Activacion',activacion)
    for solver in ("adam","lbfgs",'sgd'):
        for capa in range(2,10):
            for neuronas in range(2,15):
                red=MLPClassifier(max_iter=1500, hidden_layer_sizes=(capa,neuronas),
                                  activation=activacion, solver=solver, shuffle=False, random_state=3)
                ret=red.fit(X_train,Y_train)
                score=red.score(X_test, Y_test)
                print("Con",capa,"capas y ",neuronas,"neuronas - resultado:",score)
                if score > mejor:
                    mejor = score
                    mejorcon = (capa,neuronas,solver,activacion)

print("Score:",mejor,"con",mejorcon[0],"capas y",mejorcon[1],"neuronas, solver", mejorcon[2], 'y activacion', mejorcon[3])
"""