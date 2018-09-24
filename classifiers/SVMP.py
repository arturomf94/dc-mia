import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 

training = pd.read_csv("dat.csv")
test = pd.read_csv("test.csv")

#print(training.shape)
#print(training.head())

#print(test.shape)
#print(test.head())

xtr=training.drop('clase', axis=1)
#print(xtr)
ytr=training['clase']
#print(ytr)

xtt=test.drop('clase', axis=1)
#print(xtt)
ytt=test['clase']
#print(ytt)

svclassifier = SVC(kernel='poly', degree=9)  
svclassifier.fit(xtr, ytr)

ypr= svclassifier.predict(xtt)

print(confusion_matrix(ytt,ypr))  
print(classification_report(ytt,ypr))
cont = 0
for i in range(len(ytt)):
    if (ytt[i]==ypr[i]):
        cont = cont + 1
        #print(ytt[i],ypr[i],"éxito")
    #else:
        #print(ytt[i],ytpr[i],"error")
print("Éxitos:", cont)
print("Exactitud:", cont/len(ytt))

