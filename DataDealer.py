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
	csvReader = csv.reader(open('test.csv', encoding='utf-8'))
	column = [row for row in csvReader]
	for i in column[1:]:
		trainDict[i[0]] = i[1]
	return trainDict

#---Start From Here---
#trainDict = readTrain()
#trainIDList = []
#trainLabelList = []	#as the order of articles
corpus = []

f = open('test.json', 'r')

mw = open('train.buffer', 'w')
lines = f.readlines()
time = 0
for line in lines :
	jd = json.loads(line)
	print(jd['id'], jd['title'], jd['content'], sep = '\n')
	input()
	
	'''
	segList = jieba.cut(midString, cut_all = False)
	mw.write(' '.join(segList))
	mw.write('\n')
	'''
f.close()





