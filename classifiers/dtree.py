import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

training = pd.read_csv("datosEntrenamiento.csv")
test = pd.read_csv("datosValidacion.csv")
resultado=[]
#print(training.shape)
#print(training.head())

#print(test.shape)
#print(test.head())

xtr=training.drop('clase', axis=1)
#print(xtr)
ytr=training['clase']
#print(ytr)

#xtt=test#.drop('clase', axis=1)
#print(xtt)
#ytt=test['clase']
#print(ytt)

classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10))  
classifier.fit(xtr, ytr)

ypr= classifier.predict(test)

for j in range(644):
    resultado.append(ypr[j])
final = open('retoFiltro1.txt', mode='w')
for j in range(644):
    final.write(str(resultado[j]))
final.close

