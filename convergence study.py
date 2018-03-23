import random
import math
import csv
import itertools
import matplotlib.pyplot as plt

def preprocess(filename):
	with open(filename, 'r') as f:
		dataset = [row for row in csv.reader(f.read().splitlines())] 
	#delete rows with missing value
	for row in dataset:
		delete_row = 0
		for attribute in dataset:
			if attribute == '?':
				delete_row = 1
				break
		if(delete_row == 1):
			dataset.remove(row)		

	#pop out the original labels, don't use them in training and prediction. Only use them when calculating accuracy
	originalLabel = []
	for row in dataset:
		originalLabel.append(row.pop(-1))

	#get all unique labels	
	labels_name = []
	for label in originalLabel:
		if label not in labels_name:
			labels_name.append(label)
	#the num of label(class)
	label_num = len(labels_name)
	return (dataset, label_num, labels_name, originalLabel)


def train_unsupervised(dataset, label_num, filename):
	#assign random probability for each possible label. Every probabilty is between 0-1 and they add up to 1 
	for row in dataset:
		randomProb = []
		sumProb = 0 #use in normalization (so probabilities add up to 1)
		for i in range(0, label_num):
			a = random.uniform(0, 1)
			randomProb.append(a)
			sumProb += a
		for i in range(0, label_num):
			proba = float(randomProb[i]) / float(sumProb)
			row.append(proba)


	initialClass = []
	for row in dataset:
		classProb = row[-label_num:]
		max_value = max(classProb)
		max_index = classProb.index(max_value) 
		initialClass.append(max_index)
	x = []
	y = []
	p = []
	output = []
	N = 0 #times of iteration
	while(N <= 25):
		prob = {}
		for i in range(0, label_num): #mark the label with integer 0..label_num-1 for now. We cannot be sure what's it corresponding real label name at this stage
			prob[i] = {} #every class maintain a dictionary to keep record of probabilities
		attribute_num = len(dataset[0]) - label_num
		for i in range(attribute_num):
			for k in range(0, label_num):
				prob[k][i] = {}
				for j in range(len(dataset)):
					if(dataset[j][i] not in prob[k][i]): #a new attribute new
						prob[k][i][dataset[j][i]] = 0
					prob[k][i][dataset[j][i]] += dataset[j][-(label_num - k)]
		classProb = [0] * label_num
		for row in dataset:
			for i in range(0, label_num):
				classProb[i] += row[-(label_num - i)]
		for i in range(0, label_num):
			classProb[i] = float(classProb[i]) / float(len(dataset)) 
		#normalization
		for i in range(0, label_num):
			for attribute, attributeValue in prob[i].items():
				probSum = 0
				for key, value in attributeValue.items():
					probSum += float(value) #the total probability sum of a whole attribute
				for key, value in attributeValue.items():
					prob[i][attribute][key] = float(value) / float(probSum)
		#update label probability in dataset
		for row in dataset:
			probabilityAtt = []
			for i in range(0, label_num):
				probabilityAtt.append(classProb[i]) #initialize the list with class probability
			probAttSum = 0
			for i in range(0, len(row) - label_num):
				for j in range(0, label_num):
					probabilityAtt[j] *= prob[j][i][row[i]] #calculate the conditional probability
			for k in range(0, label_num):
				probAttSum += probabilityAtt[k] #for normalization
			for k in range(0,label_num):
				row[-(label_num - k)] = probabilityAtt[k]/probAttSum

		out = []
		newClass = []
		for row in dataset:
			classProb = row[-label_num:]
			max_value = max(classProb)
			max_index = classProb.index(max_value) 
			newClass.append(max_index)
		countDifference = 0
		for i in range(0, len(newClass)):
			if(initialClass[i] != newClass[i]):
				countDifference += 1
    	
		x.append(N)
		y.append(countDifference)
		out.append(countDifference)
		percentage = float(countDifference) / float(len(newClass))
		p.append(percentage)
		out.append(percentage)
		output.append(out)
		for i in range(0, len(newClass)):
			initialClass[i] = newClass[i]
		N += 1 #iteration times add 1
	
	with open("%s_output.csv" %filename, "w", encoding="utf8", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(output)

	plt.plot(x, y, 'ro')
	line, = plt.plot(x, y, '-')
	plt.xlabel('iteration time')
	plt.ylabel('instances change prediction')
	plt.title(filename + ": count instances change prediction")
	line.set_antialiased(False) # turn off antialising
	plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
	plt.show()

	plt.plot(x, p, 'ro')
	line, = plt.plot(x, p, '-')
	plt.xlabel('iteration time')
	plt.ylabel('percentage instances change prediction')
	plt.title(filename + ": percentage of instances change prediction")
	line.set_antialiased(False) # turn off antialising
	plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
	plt.show()
	return dataset

def predict_unsupervised(dataset, label_num):
	predict_result = []
	classProb = []
	for row in dataset:
		classProb = row[-label_num:]
		#the class with the max probability is the prediction
		maxProb = 0
		maxIndex = 0 #
		for i in range(0, label_num):
			if(classProb[i] > maxProb):
				maxProb = classProb[i]
				maxIndex = i
		predict_result.append(maxIndex)
	return predict_result

def permutation_label_name(labels_name, predict_result):
	predict_result_set = []
	allPossibleLabelPermutation = list(itertools.permutations(labels_name))
	for labelTuple in allPossibleLabelPermutation: 
		onePredictList = [] 
		for i in range(0, len(predict_result)):
			classIndex = predict_result[i]
			onePredictList.append(labelTuple[classIndex])
		predict_result_set.append(onePredictList)
	return predict_result_set		

def evaluate_unsupervised(originalLabel, predict_result_set):
	maxCorrect = 0
	finalPredictResult = []
	for predict_result in predict_result_set:
		count_correct = 0
		for i in range(0, len(originalLabel)):
			if(predict_result[i] == originalLabel[i]):
				count_correct += 1
		if(count_correct > maxCorrect):
			maxCorrect = count_correct
			finalPredictResult = predict_result #finalPredictResult is the predict result with the correct label name
	accuracy = float(maxCorrect) / float(len(originalLabel))
	return (accuracy, finalPredictResult)

def main():
	filenames = ['breast-cancer-dos.csv', 'car-dos.csv', 'hypothyroid-dos.csv', 'mushroom-dos.csv']
	for filename in filenames:
		(dataset, label_num, labels_name, originalLabel) = preprocess(filename)
		filename = filename[:-4]
		dataset = train_unsupervised(dataset, label_num, filename)
		predict_result = predict_unsupervised(dataset, label_num)
		predict_result_set = permutation_label_name(labels_name, predict_result)
		(accuracy, finalPredictResult) = evaluate_unsupervised(originalLabel, predict_result_set)
		print(accuracy)


if __name__ == '__main__':
    main()
