import random
import math
import csv
import numpy as np
def preprocess(filename):
	with open(filename, 'r') as f:
		dataset = [row for row in csv.reader(f.read().splitlines())] 
	for row in dataset:
		delete_row = 0
		for column in dataset:
			if column == '?':
				delete_row = 1
				break
		if(delete_row == 1):
			dataset.remove(row)				
	return dataset


def train_unsupervised(dataset):
	originalLabel = []
	for row in dataset:
		originalLabel.append(row.pop(-1))

	labels_name = []
	for label in originalLabel:
		if label not in labels_name:
			labels_name.append(label)

	label_num = len(labels_name)

	for row in dataset:
		randomProb = []
		sumProb = 0
		for i in range(0, label_num):
			a = random.uniform(0, 1)
			randomProb.append(a)
			sumProb += a
		for i in range(0, label_num):
			proba = float(randomProb[i]) / float(sumProb)
			row.append(proba)
		
	
	N = 0
	while(N <= 25):
		prob = {}
		for i in range(0, label_num):
			prob['class%d'%i] = {}
		column_num = len(dataset[0]) - 2
		for i in range(column_num):
			for k in range(0, label_num):
				prob['class%d'%k][i] = {}
				for j in range(len(dataset)):
					if(dataset[j][i] not in prob['class%d'%k][i]):
						prob['class%d'%k][i][dataset[j][i]] = 0
					prob['class%d'%k][i][dataset[j][i]] += dataset[j][-(label_num - k)]
		classProb = [0] * label_num
		for row in dataset:
			for i in range(0, label_num):
				classProb[i] += row[-(label_num - i)]
		for i in range(0, label_num):
			classProb[i] = float(classProb[i]) / float((len(dataset) + 1)) 
		
		for i in range(0, label_num):
			for column, columnValue in prob['class%d'%i].items():
				probSum = 0
				for key, value in columnValue.items():
					probSum += float(value)
				for key, value in columnValue.items():
					prob['class%d'%i][column][key] = float(value) / float(probSum)
		
		for row in dataset:
			probabilityAtt = []
			for i in range(0, label_num):
				probabilityAtt.append(classProb[i])
			probAttSum = 0
			for i in range(0, len(row) - label_num):
				for j in range(0, label_num):
					probabilityAtt[j] *= prob['class%d'%j][i][row[i]]
			for k in range(0, label_num):
				probAttSum += probabilityAtt[k]
			for k in range(0,label_num):
				row[-(label_num - k)] = probabilityAtt[k]/probAttSum
		N += 1
	return (label_num, originalLabel, dataset)

def predict_unsupervised(dataset, label_num):
	predict_result = []
	classProb = []
	for row in dataset:
		classProb = row[-label_num:]
		maxProb = 0
		maxIndex = 0
		for i in range(0, label_num):
			if(classProb[i] > maxProb):
				maxProb = classProb[i]
				maxIndex = i
		predict_result.append('class%d'%(maxIndex))
	return predict_result

def evaluate_unsupervised(label_num, originalLabel, predict_result):
	if(len(originalLabel) != len(predict_result)):
		print('wrong!!!')
	labels_name = []
	for label in originalLabel:
		if label not in labels_name:
			labels_name.append(label)
	
	labelCount = {}
	for i in range(0, label_num):
		labelCount['class%d'%i] = {}
		for label in labels_name:
			labelCount['class%d'%i][label] = 0
	for j in range(0, len(predict_result)):
		labelCount[predict_result[j]][originalLabel[j]] += 1
	
	count_correct = 0
	for i in range(0, label_num):
		correct_label = 0
		for key, value in labelCount['class%d'%i].items():
			if value > correct_label:
				correct_label = value
		count_correct += correct_label
	accuracy = float(count_correct) / float(len(originalLabel))
	return accuracy

def main():
	filenames = ['breast-cancer-dos.csv', 'car-dos.csv', 'hypothyroid-dos.csv', 'mushroom-dos.csv']
	for filename in filenames:
		dataset = preprocess(filename)
		label_num, originalLabel,dataset = train_unsupervised(dataset)
		predict_result = predict_unsupervised(dataset, label_num)
		accuracy = evaluate_unsupervised(label_num, originalLabel, predict_result)
		print(accuracy)


if __name__ == '__main__':
    main()
