infile = open('train.json', 'r')
outfile = open('train_sample.json', 'w')

for i in range(3):
	outfile.writelines(infile.readline().strip())