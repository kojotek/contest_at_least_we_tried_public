#  -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from collections import defaultdict
from sklearn.linear_model import ElasticNetCV

#reading data
print("Loading Data...")
data = list()
with open('../train/train.tsv','r', encoding="utf8") as file:
	tsvin = csv.reader(file, delimiter='\t')
	for row in tsvin:
		data.append(row)

dataZipped = zip(*data)
columns = list(dataZipped)

prices = np.array(list(map(float, columns[0])))
rooms = np.array(list(map(float, columns[1])))
meters = np.array(list(map(float, columns[2])))
floors = np.array(list(map(float, columns[3])))
pricesPerMeter = [a/b for a,b in zip(prices,meters)]


#reading coords
coordData = list()
with open('../coords.tsv','r', encoding="utf8") as file:
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

def rejectOutliers(data, m):
	matching = set();
	for i in range(len(data)):
		if (abs(data[i] - np.mean(data)) < m * np.std(data)):
			matching.add(i)
	return matching

	
normalizationParams = [0.8, 2, 2, 2, 1, 1.4, 1.4, 1]

print ("Removing outliers with parameters {0}...".format(normalizationParams))

match = set.intersection( 
	rejectOutliers(prices, normalizationParams[0]),
	rejectOutliers(rooms, normalizationParams[1]),
	rejectOutliers(meters, normalizationParams[2]),
	rejectOutliers(floors, normalizationParams[3]),
	rejectOutliers(pricesPerMeter, normalizationParams[4]),
	rejectOutliers(firstCoords, normalizationParams[5]),
	rejectOutliers(secondCoords, normalizationParams[6]),
	rejectOutliers(distanceCenter, normalizationParams[7]))

pricesFiltered = [prices[i] for i in match]
roomsFiltered = [rooms[i] for i in match]
metersFiltered = [meters[i] for i in match]
floorsFiltered = [floors[i] for i in match]
pricesPerMeterFiltered = [pricesPerMeter[i] for i in match]
#wuyrzucimy moze w ten sposob bledy i dalekie peryferie miasta
firstCoordsFiltered = [firstCoords[i] for i in match]
secondCoordsFiltered = [secondCoords[i] for i in match]
distanceCenterFiltered = [distanceCenter[i] for i in match]

print ("Outliers removed! Remaining {0} out of {1} items.".format(len(match), len(prices)))


#tu była ambinta proba obuczenia sieci nauronowej, niestety, zakonczona porazka
##skalowanie coordow - z jakiegos powodu sklearn chce sie uczyc tylko na intach, wiec odpowiednie przeskalowanie modelu powinno wystarczyc
#print ("Generating neural network for Coordintes - PricePerMeter relation")
#
#firstCoordsMultiplied = [x*1000000 for x in firstCoordsFiltered]
#secondCoordsMultiplied = [x*1000000 for x in secondCoordsFiltered]
#pricesPerMeterMultiplied = [x for x in pricesPerMeterFiltered]
#
#neuralX = np.array(list(zip(firstCoordsMultiplied, secondCoordsMultiplied)))
#neuralY = pricesPerMeterMultiplied
#neuralXuniques = list()
#neuralYuniques = list()
#
#print ("Searching for unique coords for neural learning...")
#setX = set(zip(firstCoordsMultiplied, secondCoordsMultiplied))
#print ("{0} unique values found".format(len(setX)))
#
#uniqueX = list(setX)
#
#for u in uniqueX:
#	meanElements = np.array([neuralY[i] for i in range(len(neuralY)) if list(neuralX[i]) == list(u)])
#	mean = meanElements.mean()
#	neuralXuniques.append(u)
#	neuralYuniques.append(mean)
#
#neuralXuniques = np.array(neuralXuniques).astype(int)
#neuralYuniques = np.array(neuralYuniques).astype(int)
#
#neuralNetwork = MLPClassifier(hidden_layer_sizes=(5,3), max_iter=100000, alpha=0.001,
#                    solver='sgd', verbose=10, tol=1, random_state=1,
#                    learning_rate_init=0.01)
#
#
#neuralNetwork.fit(neuralXuniques, neuralYuniques)
#
#neuralFile = "nnet.pkl"
#print("Neural network generated!")
#print("Saving network in file{0}...".format(neuralFile))
#joblib.dump(neuralNetwork, neuralFile)
#print("Neural network saved!")
#
##nie jestem pewien, czy to dobry pomysl, ale warto sprobowac
#localizationFactor = neuralNetwork.predict(neuralX).astype(float)
#print(localizationFactor)


print ("Normalizing...")

filteredData = np.array(list(zip(roomsFiltered, metersFiltered, floorsFiltered, distanceCenterFiltered)))

scaler = preprocessing.StandardScaler().fit(filteredData)

normFile = 'norm.pkl'
print ("Saving normalization parameters in file{0}...".format(normFile))
joblib.dump(scaler, normFile)
print ("normalization parameters saved!")

scaledData = scaler.transform(filteredData)

print ("Normalization completed!")


X = scaledData
y = pricesPerMeterFiltered

print ("Creating regression...")

elasticNet = linear_model.ElasticNetCV(
	l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
	fit_intercept=True, normalize=False, precompute='auto',
	max_iter=1000, tol=0.0001, cv=None, copy_X=True,
	verbose=0, n_jobs=1, positive=False, random_state=None,
	selection='cyclic')

elasticNet.fit(X, y)

regFile = "reg.pkl"
print ("Saving regression object in file{0}...".format(regFile))
joblib.dump(elasticNet, regFile)
print ("Regression object saved!")


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