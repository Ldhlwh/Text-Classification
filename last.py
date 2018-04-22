import xgboost as xgb

bst = xgb.Booster()
bst.load_model('0417-1-0.8.model')

dtest = xgb.DMatrix('testDMatrix.buffer')

output = open('prediction-0417-1-0.8-last.data', 'w')
print('---Start to Predict---')
ypred = bst.predict(dtest)
print('---Wrinting the Outcome---')
for line in ypred:
	output.write(str(line))
	output.write('\n')
output.close()