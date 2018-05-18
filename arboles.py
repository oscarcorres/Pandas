# 1.- Crea un fichero llamado arboles.py
#
# 2.- Carga los datos de cancer

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pydotplus
import collections


cancer=datasets.load_breast_cancer()
X=cancer.data
Y=cancer.target

# 3.- Aplica el algoritmo de Árboles de decisión

#Dividimos en entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)


#Entrenamos el algoritmo con gini

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
print(clf_gini)

#Entrenamos el algoritmo con entropy
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# 4.- Comprobar su efectividad

#Predecimos con gini
y_pred_gini = clf_gini.predict(X_test)

#Acierto con gini
print ("Acierto con gini es ", accuracy_score(y_test,y_pred_gini)*100)


#Predecimos con entropy
y_pred_en = clf_entropy.predict(X_test)
#

#Acierto con entropy
print ("Acierto con entropy es ", accuracy_score(y_test,y_pred_en)*100)

# 5.- Generar el fichero .dot y la gráfica

#Salvando a fichero .dot
dotfile = open("./figures/arbol.dot", 'w')
feature_names=cancer.feature_names
dot_data=export_graphviz(clf_gini, out_file = dotfile, feature_names = feature_names)
dotfile.close()

dotfile = open("./figures/arbol2.dot", 'w')
dot_data=export_graphviz(clf_entropy, out_file = dotfile, feature_names = feature_names)
dotfile.close()

#Salvando a fichero png
def genera_png(tree, feature_names, filepath):
    colors = ('turquoise', 'orange')
    dot_data=export_graphviz(tree, out_file = None, feature_names = feature_names,filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(filepath)

genera_png(clf_gini,feature_names,'./figures/tree.png')
genera_png(clf_entropy,feature_names,'./figures/tree2.png')

