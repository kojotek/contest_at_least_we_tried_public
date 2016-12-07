import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from collections import defaultdict
from sklearn.linear_model import ElasticNetCV
import sys
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from math import sqrt

import contestLib as cl


TARGET = "gonito"


###to mozna modyfikowac do woli
___poznanCenter = [52.407860, 16.928249]

___removalParams = [2, 2, 2, 2, 2, 2, 2, 2, 2, 100, 100, 100, 100, 100, 100, 100]


___learningDataBool = [
	0,	#prices##NEVER!
	1,	#rooms
	1,	#meters
	0,	#floors
	0,	#pricesPerMeterNEVER!
	0,	#first coords
	0,	#second coords
	0,	#distance
	0,  #metersPerRoom
	1,  #flatType
	1,  #condition
	0,  #protected
	0,  #parking
	0,  #kitchen
	0,  #garden
	0  #hardLocalization
	]

#huber 364.17868, eps_ins = 363.66310, sq_eps_ins = 370.294969
___lossMethod = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
___lossMethodN = 2

#none = 363.66310, l2 = 363.61123,l 1 = 363.81478077037843, elasticnet = 363.3668235609241
___penaltyMethod = ['none', 'l2', 'l1', 'elasticnet']
___penaltyMethodN = 3

___shuffleSeed = 0

#invscaling da best
___learningRate = ['constant', 'optimal', 'invscaling']
___learningRateN = 1

___regression = linear_model.SGDRegressor(
		alpha=0.0001,
		average=False,
		epsilon=0.01,
		eta0=0.01,
		fit_intercept=True,
		l1_ratio=0.4,
		learning_rate=___learningRate[___learningRateN],
		loss=___lossMethod[___lossMethodN], 
		n_iter=10000,
		penalty=___penaltyMethod[___penaltyMethodN],
		power_t=0.25,
		random_state=___shuffleSeed,
		shuffle=True,
		verbose=1,
		warm_start=False
		)

#linear_model.ElasticNetCV(
#	l1_ratio=0.5, eps=0.0001, n_alphas=100, alphas=None,
#	fit_intercept=True, normalize=False, precompute='auto',
#	max_iter=10000, tol=0.0001, cv=None, copy_X=True,
#	verbose=0, n_jobs=1, positive=False, random_state=None,
#	selection='cyclic')

___predicted = 0 #0 is price, 1 is for price per meter
####



prices, rooms, meters, floors, pricesPerMeter, locations, descriptions = cl.loadTrainData()
firstCoords, secondCoords = cl.loadTrainCoords()

___distanceCenter = cl.getDistanceToPoint(firstCoords, secondCoords, ___poznanCenter[0], ___poznanCenter[1])

metersPerRoom = [m/r for m,r in zip(meters,rooms)]

flatType = cl.getFlatType(descriptions)
condition = cl.getCondition(descriptions)
protected = cl.getProtected(descriptions)
parking = cl.getParking(descriptions)
kitchen = cl.getKitchen(descriptions)
garden = cl.getGarden(descriptions)
hardLocalization = cl.getHardLocalization(descriptions)

	
filteredData = cl.removeOutliers([prices, rooms, meters, floors, pricesPerMeter, firstCoords, secondCoords, ___distanceCenter, metersPerRoom, flatType, condition, protected, parking, kitchen, garden, hardLocalization], ___removalParams)

filtered_prices = filteredData[0]
filtered_pricesPerMeter = filteredData[4]

___learningData = cl.compress(filteredData, ___learningDataBool)


#for i in range(len(dataList)):
#	if ___learningDataBool[i] >= 1:
#		___learningData.append(dataList[i])


#scaler = cl.generateScaler(___learningData)
#scaledData = cl.scaleData(___learningData, scaler)


expected = list()

if ___predicted >= 1:
	expected = filtered_pricesPerMeter
else:
	expected = filtered_prices


#model = cl.createRegression(scaledData, expected, ___regression)
model = cl.createRegression(___learningData, expected, ___regression)


###########predykcja i ocena bledu

loadDataFunc = None

if TARGET == 'gonito':
	loadDataFunc = [cl.loadTestData, cl.loadTestCoords]
else:
	loadDataFunc = [cl.loadDevData, cl.loadDevCoords]

test_rooms, test_meters, test_floors, test_locations, test_descriptions = loadDataFunc[0]()
test_firstCoords, test_secondCoords = loadDataFunc[1]()
test_distanceCenter = cl.getDistanceToPoint(test_firstCoords, test_secondCoords, ___poznanCenter[0], ___poznanCenter[1])

test_metersPerRoom = [m/r for m,r in zip(test_meters,test_rooms)]

test_flatType = cl.getFlatType(test_descriptions)
test_condition = cl.getCondition(test_descriptions)
test_protected = cl.getProtected(test_descriptions)
test_parking = cl.getParking(test_descriptions)
test_kitchen = cl.getKitchen(test_descriptions)
test_garden = cl.getGarden(test_descriptions)
test_hardLocalization = cl.getHardLocalization(test_descriptions)

test_dataList = [[], test_rooms, test_meters, test_floors, [], test_firstCoords, test_secondCoords, test_distanceCenter, test_metersPerRoom, test_flatType, test_condition, test_protected, test_parking, test_kitchen, test_garden, test_hardLocalization]

test_data = cl.compress(test_dataList, ___learningDataBool)

predicted = cl.predict(test_data, model)

if TARGET == 'train':
	predicted = cl.predict(___learningData, model)

if ___predicted >= 1:
	predicted = [a*b for a,b in zip(predicted,test_meters)]

test_expected = None

#mn = 511.
#predicted = [mn + np.sign(p-mn)*np.sqrt(np.abs(p-mn)) for p in predicted ]
	
if TARGET == 'gonito':
	for p in predicted:
		print('{0:f}'.format(p))	
elif TARGET == 'dev':
	test_expected = cl.loadDevExpected()
	print(cl.rmse(test_expected, predicted))
elif TARGET == 'train':
	test_expected = filtered_prices
	print(cl.rmse(test_expected, predicted))
