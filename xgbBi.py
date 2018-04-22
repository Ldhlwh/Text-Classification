import xgboost as xgb

print('---Loading dtrain---')
dtrain = xgb.DMatrix('trainBiDMatrixClip.buffer')
print('---Loading dtest---')
dtest = xgb.DMatrix('testBiDMatrix.buffer')
print('---Loading dval---')
dval = xgb.DMatrix('valBiDMatrix.buffer')

#---Parameter---
param = {'eta' : 0.01,
	'max_depth' : 11,
	'objective' : 'binary:logistic',
	'eval_metric' : 'auc',
	'silent' : True,
	'min_child_weight' : 2,
	'sub_sample' : 1.0,
	'colsample_bytree' : 0.8
	}
numRound = 100000
watchlist = [(dtrain, 'train'), (dval, 'val')]

#---Training---
print('---Start to train---')
bst = xgb.train(param, dtrain, numRound, early_stopping_rounds = 500, evals = watchlist, verbose_eval = True)
print('---Saving Model---')
bst.save_model('0421-1.model')
#---Predicting and Output---
output = open('prediction-0421-1.data', 'w')
print('---Start to Predict---')
ypred = bst.predict(dtest, ntree_limit = bst.best_iteration)
print('---Wrinting the Outcome---')
for line in ypred:
	output.write(str(line))
	output.write('\n')
output.close()

output = open('prediction-0421-1-last.data', 'w')
print('---Start to Predict---')
ypred = bst.predict(dtest)
print('---Wrinting the Outcome---')
for line in ypred:
	output.write(str(line))
	output.write('\n')
output.close()









	





