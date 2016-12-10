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
import sys
from copy import deepcopy
import re
from words import program
import pickle

from sklearn.metrics import mean_squared_error
from math import sqrt


def rmse(predicted, actual):
    return np.sqrt(((predicted - actual) ** 2).mean())


def loadTrainData():
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
	locations = columns[4]
	descriptions = columns[5]
	pricesPerMeter = [a/b for a,b in zip(prices,meters)]
	return prices, rooms, meters, floors, pricesPerMeter, locations, descriptions


def loadTrainPrices():
	#reading coords
	prices = list()
	with open('../train/prices.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			prices.append(float(row[0]))

	return prices
	

def loadTrainCoords():
	#reading coords
	coordData = list()
	with open('../train/coords.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			coordData.append(row)

	columnsZipped = zip(*coordData)
	coordColumns = list(columnsZipped);

	firstCoords = np.array(list(map(float, coordColumns[0])))
	secondCoords = np.array(list(map(float, coordColumns[1])))
	return firstCoords, secondCoords


def loadDevData():
	data = list()
	with open('../dev-0/in.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			data.append(row)

	dataZipped = zip(*data)
	columns = list(dataZipped)

	rooms = np.array(list(map(float, columns[0])))
	meters = np.array(list(map(float, columns[1])))
	floors = np.array(list(map(float, columns[2])))
	locations = columns[3]
	descriptions = columns[4]
	return rooms, meters, floors, locations, descriptions


def loadDevCoords():
	#reading coords
	coordData = list()
	with open('../dev-0/coords.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			coordData.append(row)

	columnsZipped = zip(*coordData)
	coordColumns = list(columnsZipped);

	firstCoords = np.array(list(map(float, coordColumns[0])))
	secondCoords = np.array(list(map(float, coordColumns[1])))
	return firstCoords, secondCoords

def loadDevExpected():
	data = list()
	with open('../dev-0/expected.tsv','r', encoding="utf8") as file:
		for row in file:
			data.append(row)

	data2 = np.array(list(map(float, data)))
	return data2
	
def loadTestData():
	data = list()
	with open('../test-A/in.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			data.append(row)

	dataZipped = zip(*data)
	columns = list(dataZipped)

	rooms = np.array(list(map(float, columns[0])))
	meters = np.array(list(map(float, columns[1])))
	floors = np.array(list(map(float, columns[2])))
	locations = columns[3]
	descriptions = columns[4]
	return rooms, meters, floors, locations, descriptions


def loadTestCoords():
	#reading coords
	coordData = list()
	with open('../test-A/coords.tsv','r', encoding="utf8") as file:
		tsvin = csv.reader(file, delimiter='\t')
		for row in tsvin:
			coordData.append(row)

	columnsZipped = zip(*coordData)
	coordColumns = list(columnsZipped);

	firstCoords = np.array(list(map(float, coordColumns[0])))
	secondCoords = np.array(list(map(float, coordColumns[1])))
	return firstCoords, secondCoords


def getDistanceToPoint(firstCoords, secondCoords, xb, yb):
	distanceCenter = list()

	for i in range(len(firstCoords)):
		xa = firstCoords[i]
		ya = secondCoords[i]
		dist = np.sqrt((xa-xb)**2 + (ya-yb)**2)
		distanceCenter.append(dist)
	return distanceCenter


def checkRegex(dictionary, data, default=-1):
	result = list()
	for row in data:
		result.append(default)
		for r in dictionary.keys():
			regex = re.compile(r,re.IGNORECASE)
			match = regex.search(row)
			if match is not None:
				result[-1] = dictionary[r]
	return result


#def getFlatType(descriptions):
#	return checkRegex({
#			'.*kawalerk.*':1,
#			'.*studio.*':1,
#			'.*blok.*':2, 
#			'.*niskim? blok.*':3,
#			'.*apartament.*':5,
#			'.*apartamentow.*':4,
#			'.*jednorodz.*':6,
#			'.*szeregow.*':6,
#			'.*blizniak.*':6},
#		descriptions,
#		default=0)
#
#
#def getCondition(descriptions):
#	return checkRegex(
#	{
#		'.*do.*?remontu.*':1,
#		'.*do.*?wykoncze.*':2,
#		'.*bez wykoncz.*':2,
#		'.*odrestauro.*':3,
#		'.*rewital.*':4, 
#		'.*odnowio.*':4,
#		'.*po.*?remoncie.*':5,
#		'.*wyremontow.*':5,
#		'.*pod klucz.*':6,
#		'.*peln.*?wyposaz.*':6},
#	descriptions,
#	default=0)
#
#	
#def getProtected(descriptions):
#	return checkRegex(
#	{
#		'.*monitor.*':1,
#		'.*strzezo.*':1,
#		'.*ochrona.*':1,
#		'.*chronion.*':1,
#		},
#	descriptions,
#	default=0)
#
#
#def getParking(descriptions):
#	return checkRegex(
#	{
#		'.*parking.*':1,
#		'.*postoj.*':1,
#		'.*garaz.*':2
#		},
#	descriptions,
#	default=0)
#
#
#def getKitchen(descriptions):
#	return checkRegex(
#	{
#		'.*aneks.*':1,
#		'.*kuchni.*':2
#		},
#	descriptions,
#	default=1)
#
#
#def getHardLocalization(descriptions):
#	return checkRegex(
#	{
#		'.*now.*?osiedl.*':1,
#		'.*taras.*?warty*':2
#		},
#	descriptions,
#	default=0)	
#
#
#def getGarden(descriptions):
#	return checkRegex(
#	{
#		'.*ogrodek.*':1,
#		'.*ogrodkiem.*':1,
#		'.*dzialk.*':1
#		},
#	descriptions,
#	default=0)
	

def _rejectOutliers__(data, m):
	matching = set();
	for i in range(len(data)):
		if (abs(data[i] - np.mean(data)) < m * np.std(data)):
			matching.add(i)
	return matching


def removeOutliers(data, normalizationParams):
	sets = list()
	for i in range(len(data)):
		sets.append(_rejectOutliers__(data[i], normalizationParams[i]))
		
	match = set.intersection(*sets)
	filteredData = list()
	
	for d in data:
		filteredData.append([d[i] for i in match])
	
	return filteredData


def generateScaler(data):

	filteredData = np.array(list(zip(*data)))
	scaler = preprocessing.StandardScaler().fit(filteredData)
	#scaledData = scaler.transform(filteredData)
	
	return scaler

def scaleData(data, scaler):
	scaledData = scaler.transform(np.array(list(zip(*data))))
	return np.array(list(zip(*scaledData)))
	
def createRegression(data, expected, regressor):
	reg2 = deepcopy(regressor)
	reg2.fit(np.array(list(zip(*data))), expected)
	return reg2

def predict(data, regresor):
	pred = regresor.predict(np.array(list(zip(*data))))
	return pred
	
def compress(data, selectors):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return [d for d, s in zip(data, selectors) if s]
	
def saveObj(obj, name ):
	with open(name, 'w', encoding="utf8", newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in obj:
			spamwriter.writerow([i[0], i[1]])
        

def loadObj(name ):
	lst = list()
	with open(name, 'r', encoding="utf8", newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
		for row in spamreader:
			lst.append([float(row[0]), row[1]])
	return lst