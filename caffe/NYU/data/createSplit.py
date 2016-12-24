#!/home/zhujun/python27/bin/python
## ./createSplit.py path .jpg  

import sys
import os
from os import walk
import scipy.io as scio
import random

'''
This file creates train.txt and test.txt
(in which each line is just filename) 
for python layer's need.
'''

allnames = []
data_path = sys.argv[1]
suffix = sys.argv[2]

for dirpath, dirnames, filenames in walk(data_path):
	for filename in filenames:
		allnames.append(filename[:-len(suffix)])

random.shuffle(allnames)

train_num = len(allnames)*4/5
trainval = allnames[:train_num]
test = allnames[train_num:]

with open('train.txt', 'w') as f:
	for x in trainval:
		f.write(x+'\n')

with open('test.txt', 'w') as f:
	for x in test:
		f.write(x+'\n')
