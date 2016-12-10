#  -*- coding: utf-8 -*-


import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from collections import defaultdict
from sklearn.linear_model import ElasticNetCV

#jako pierwszy argument podaj sciezke pliku z danymi, jako drugi argument podaj sciezke do pliku z coordami

#reading data
print("Loading Data...")
data = list()
with open(sys.argv[1],'r', encoding="utf8") as file:
	tsvin = csv.reader(file, delimiter='\t')
	for row in tsvin:
		data.append(row)

dataZipped = zip(*data)
columns = list(dataZipped)

rooms = np.array(list(map(float, columns[0])))
meters = np.array(list(map(float, columns[1])))
floors = np.array(list(map(float, columns[2])))


#reading coords
coordData = list()
with open(sys.argv[2],'r', encoding="utf8") as file:
	tsvin = csv.reader(file, delimiter='\t')
	for row in tsvin:
		coordData.append(row)

columnsZipped = zip(*coordData)
coordColumns = list(columnsZipped);

firstCoords = np.array(list(map(float, coordColumns[0])))
secondCoords = np.array(list(map(float, coordColumns[1])))

distanceCenter = list()

for i in range(len(firstCoords)):
	xa = firstCoords[i]
	ya = secondCoords[i]
	xb = 52.407860
	yb = 16.928249
	dist = np.sqrt((xa-xb)**2 + (ya-yb)**2)
	distanceCenter.append(dist)


print("Data loaded!")


print ("Normalizing...")

loadedData = np.array(list(zip(rooms, meters, floors, distanceCenter)))

normFile = 'norm.pkl'
print ("Saving normalization parameters in file{0}...".format(normFile))
scaler = joblib.load(normFile)
print ("normalization parameters saved!")

scaledData = scaler.transform(loadedData)

print ("Normalization completed!")



regFile = "reg.pkl"
print ("Loading regression object from file{0}...".format(regFile))
elasticNet = joblib.load(regFile)
print ("Regression object loaded!")

X = scaledData

prediction = elasticNet.predict(X)
predictedPrices = [prediction[i]*meters[i] for i in range(len(prediction))]

predFile = "pred.txt"
print ("Saving predicted data in file{0}...".format(predFile))

stringedPrices = '\n'.join(str(e) for e in predictedPrices)

with open(predFile, 'w') as file_:
    file_.write(stringedPrices)

print ("Predicted data saved!")

## Plot the decision boundary. For that, we will assign a color to each
## point in the mesh [x_min, x_max]x[y_min, y_max].
#x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#Z = elasticNet.predict(np.c_[xx.ravel(), yy.ravel()])
#
## Put the result into a color plot
#Z = Z.reshape(xx.shape)
#plt.figure(1, figsize=(4, 3))
#plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
## Plot also the training points
#plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#
#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())
#plt.xticks(())
#plt.yticks(())
#
#plt.show()