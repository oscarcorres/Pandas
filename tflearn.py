# Crea un fichero llamado mlpc.py
# Carga los datos de cancer de sklearn

from sklearn import datasets
import numpy as np
import tflearn
from tflearn import data_utils as dt
import tensorflow

cancer=datasets.load_breast_cancer()
#print(cancer)
X=cancer.data
Y=cancer.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,random_state=8)
X_train =np.array(X_train, dtype=np.float32)
Y_train = dt.to_categorical(Y_train,2)
X_test =np.array(X_test, dtype=np.float32)
Y_test = dt.to_categorical(Y_test,2)
print(Y.shape)

net = tflearn.input_data(shape=[None,30])
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
print("Fitting")
model.fit(X_train, Y_train, n_epoch=100,batch_size=32,
          show_metric=False, snapshot_epoch=False, snapshot_step=False,
          shuffle=False)
print("Evaluate")
print(model.evaluate(X_test,Y_test,batch_size=128))
"""
resultado = []
for nodo1 in range(16,31):
    for nodo2 in range(9,nodo1 + 1):
        print("nodo1:", nodo1)
        print("nodo2:", nodo2)

        nets = tflearn.input_data(shape=[None, 30]))
        nets= tflearn.fully_connected(nets[i], nodo1)
        nets = tflearn.fully_connected(nets[i], nodo2)
        nets = tflearn.fully_connected(nets[i], 2, activation='softmax')
        nets = tflearn.regression(nets[i])


        models.append( tflearn.DNN(nets[i]))
        print("Fitting")
        models[i].fit(X_train, Y_train, n_epoch=1, batch_size=32,
                  show_metric=False, snapshot_epoch=False, snapshot_step=False,
                  shuffle=False)
        print("Evaluate")
        evaluacion=models[i].evaluate(X_test, Y_test, batch_size=128)
        print(evaluacion)
        resultado = resultado + [[nodo1, nodo2, evaluacion]]

print(resultado)

"""
