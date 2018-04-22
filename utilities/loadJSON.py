import json

with open('train.json', 'r') as file:
	lines = file.readlines()
	for line in lines :
		temp = json.loads(line)
		