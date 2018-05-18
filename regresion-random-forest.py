# 6.- Genera un fichero llamado regresion-random-forest.py
#
# 7.- Carga los datos de boston

rom sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pydotplus
import collections


cancer=datasets.load_boston()
X=cancer.data
Y=cancer.target

# 8.- Utiliza el algortimo de random forest



# 9.- Evalua su eficacia
#
# 10.- genera un png con un uno de los arboles generados para estimar