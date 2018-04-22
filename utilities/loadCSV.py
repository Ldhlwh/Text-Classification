import csv

def readTrain():
	trainDict = {}
	csvReader = csv.reader(open('train.csv', encoding='utf-8'))
	column = [row for row in csvReader]
	for i in column[1:]:
		trainDict[i[0]] = i[1]
	return trainDict
	
train = readTrain()
