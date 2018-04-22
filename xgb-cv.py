import json
import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
import xgboost as xgb

def isChinese(ch):
	if ch >= u'\u4e00' and ch <= u'\u9fa5':
		return True
	return False
	
def readTrain():
	trainDict = {}
	csvReader = csv.reader(open('train.csv', encoding='utf-8'))
	column = [row for row in csvReader]
	for i in column[1:]:
		trainDict[i[0]] = i[1]
	return trainDict

#---Start From Here---
#trainDict = readTrain()
#trainIDList = []
trainLabelList = []	#as the order of articles
corpus = []
testCorpus = []

'''	#read train.buffer
print('---Reading Training Data---')
bf = open('train.buffer', 'r')	#Read Training Data
lines = bf.readlines()
for line in lines:
	corpus.append(line.strip('\n'))
bf.close()
'''

print('---Reading Training Label---')
lbf = open('trainLabel.buffer', 'r')	#Read Training Label
lines = lbf.readlines()
for line in lines:
	trainLabelList.append(float(line.strip('\n')))
lbf.close()

'''	#read test.buffer
print('---Reading Test Data---')
tbf = open('test.buffer', 'r')	#Read Testing Data
lines = tbf.readlines()
for line in lines:
	testCorpus.append(line.strip('\n'))
tbf.close()
'''

'''	#vector fit
print('---Begin to fit---')
vectorizer = TfidfVectorizer(min_df = 1)
vec = vectorizer.fit_transform(corpus + testCorpus)
'''
	
#---DMatrix---
''' #make and save dtrain and dtest
print('---Making dtrain---')
dtrain = xgb.DMatrix(vec[:len(corpus)], label = trainLabelList)
print('---Saving dtrain---')
dtrain.save_binary('trainDMatrix.buffer')

print('---Making dtest---')
dtest = xgb.DMatrix(vec[len(corpus):])
print('---Saving dtest---')
dtest.save_binary('testDMatrix.buffer')
'''

print('---Loading dtrain---')
dtrain = xgb.DMatrix('trainDMatrix.buffer')
#dtrain = dtrain.slice(list(range(1000, 321910)))
#dtrain.save_binary('trainDMatrixClip.buffer')
#dtrain = xgb.DMatrix('trainDMatrixClip.buffer')
print('---Loading dtest---')
dtest = xgb.DMatrix('testDMatrix.buffer')

''' #make and save dval
dvalList = list(range(1000))
print('---Making dval---')
dval = dtrain.slice(dvalList)
print('---Saving dval---')
dval.save_binary('valDMatrix.buffer')
'''

'''	#load dval
print('---Loading dval---')
dval = xgb.DMatrix('valDMatrix.buffer')
'''

#---Parameter---
param = {'eta' : 0.1,
	'max_depth' : 15,
	'objective' : 'binary:logistic',
	'eval_metric' : 'auc',
	'silent' : True,
	'min_child_weight' : 2
	}
numRound = 10000
#watchlist = [(dtrain, 'train'), (dval, 'val')]

#---CV---
kFold = 5
print('---Start to Cross Validation---')
import time
bst = xgb.cv(param, dtrain, numRound, kFold, stratified = True, verbose_eval = True)


	





