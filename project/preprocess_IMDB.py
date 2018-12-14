import numpy as np
import os
import nltk
import itertools
import io

def import_data():

	## get all of the training reviews (including unlabeled reviews)
	train_directory = 'aclImdb/train/'
	pos_filenames = os.listdir(train_directory + 'pos/')
	neg_filenames = os.listdir(train_directory + 'neg/')
	unsup_filenames = os.listdir(train_directory + 'unsup/')
	pos_filenames = [train_directory+'pos/'+filename for filename in pos_filenames]
	neg_filenames = [train_directory+'neg/'+filename for filename in neg_filenames]

	filenames = pos_filenames + neg_filenames
	count = 0
	neg_train = []
	pos_train = []
	for filename in filenames:
		with io.open(filename,'r',encoding='utf-8') as f:
			line = f.readlines()[0]
		line = line.replace('<br />',' ')
		line = line.replace('\x96',' ')
		
		if filename in pos_filenames:
			pos_train.append(line)
		else:
			neg_train.append(line)
		count += 1
		#print(count)

	## get all of the test reviews
	test_directory = 'aclImdb/test/'
	pos_filenames = os.listdir(test_directory + 'pos/')
	neg_filenames = os.listdir(test_directory + 'neg/')
	pos_filenames = [test_directory+'pos/'+filename for filename in pos_filenames]
	neg_filenames = [test_directory+'neg/'+filename for filename in neg_filenames]
	filenames = pos_filenames+neg_filenames
	count = 0
	pos_test = []
	neg_test = []
	for filename in filenames:
		with io.open(filename,'r',encoding='utf-8') as f:
			line = f.readlines()[0]
		line = line.replace('<br />',' ')
		line = line.replace('\x96',' ')

		if filename in pos_filenames:
			pos_test.append(line)
		else:
			neg_test.append(line)
		count += 1
		#print(count)

	return neg_train, pos_train, neg_test, pos_test

