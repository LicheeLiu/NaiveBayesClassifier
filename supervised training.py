#python3 is used for this program

import csv
import math
import random
import numpy as np

def preprocess(filename):
	with open(filename, 'r') as f:
		dataset = [row for row in csv.reader(f.read().splitlines())] 
	return dataset

#split dataset into training data and test data; the split ratio is set in main(), but where each instance goes is random
def splitDataSet(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def train_supervised(dataset):
    #separate the whole dataset by class. The class value is in the last column of every row.
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated): #a new class
            separated[vector[-1]] = []  
        separated[vector[-1]].append(vector) #append the vector into the class it belongs
    attributesNum = len(dataset[0]) - 1

    #count occurence of every attributes and count total number of class
    #skip when thre is '?' (missing data)
    countAttributes = {}
    countTotal = {}
    for i in range(0, attributesNum):
        countAttributes[i] ={}
        for classValue, classVectors in separated.items():
            countAttributes[i][classValue] = {}
            nonMissingTotal = 0
            for vector in classVectors:
                #don't count in missing data 
                attributeValue = vector[i]
                if(attributeValue == '?'):
                    break
                if vector[i] not in countAttributes[i][classValue]:
                    countAttributes[i][classValue][attributeValue] = 0
                countAttributes[i][classValue][attributeValue] += 1
                nonMissingTotal += 1
            for attributeValue, attributeCount in countAttributes[i][classValue].items():
                countAttributes[i][classValue][attributeValue] = float(attributeCount) / float(nonMissingTotal)
            countAttributes[i][classValue]['others'] = np.nextafter(0, 1)   #for attributes not existed, assign it to a small positive number for smoothing
    return countAttributes 

#remove instance with '?' in testset
def missingDataTestSet(testSet):
    for vector in testSet:
        missingData = 0
        for attribute in vector:
            if(attribute == '?'):
                missingData = 1
                break
        if(missingData == 1):
            testSet.remove(vector)
    return testSet

def predict_supervised(testSet, countAttributes, dataset):
    allClass = []
    #keep record of all unique class names
    for row in dataset:
        if row[-1] not in allClass:
            allClass.append(row[-1])

    prediction = []
    #call predict_line() function to predict each instance in the testSet
    for testVector in testSet:
        prediction.extend(predict_line(allClass, testVector, countAttributes))
    return prediction

def predict_line(allClass, testVector, countAttributes):
    logProbaility = {}
    for oneClass in allClass:
        logProbaility[oneClass] = 0
        for i in range(0, len(testVector)-1):
            if testVector[i] in countAttributes[i][oneClass]:
                #use log probability to avoid underflow
                logProbaility[oneClass] += math.log(countAttributes.get(i).get(oneClass).get(testVector[i]))
            else:
                logProbaility[oneClass] += math.log(countAttributes.get(i).get(oneClass).get('others'))
    max_value = max(logProbaility.values())
    max_key = [k for k, v in logProbaility.items() if v == max_value] #the prediction is the max probability among all classes
    return max_key

def evaluate_supervised(prediction, testSet):
    countCorrect = 0
    for i in range(0, len(prediction)):
        if(prediction[i] == testSet[i][-1]):
            countCorrect += 1
    return float(countCorrect) / float(len(prediction))    

def main():
    filenames = ['breast-cancer-dos.csv', 'car-dos.csv', 'hypothyroid-dos.csv', 'mushroom-dos.csv']
    for filename in filenames:
        splitRatio = 0.8
        dataset = preprocess(filename)
        trainingSet, testSet = splitDataSet(dataset, splitRatio)
        countAttributes = train_supervised(trainingSet)
        testSet = missingDataTestSet(testSet)
        prediction = predict_supervised(testSet, countAttributes, dataset)
        accuracy = evaluate_supervised(prediction, testSet)
        print(accuracy)


if __name__ == '__main__':
    main()
