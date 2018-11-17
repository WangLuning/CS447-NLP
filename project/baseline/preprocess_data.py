import numpy as np
import os
import nltk
import itertools
import io
## create directory to store preprocessed data
if(not os.path.isdir('../preprocessed_data')):
	os.mkdir('../preprocessed_data')

train_directory = ''
pos_filenames = ['../rt-polarity.pos']
neg_filenames = ['../rt-polarity.neg']
filenames = pos_filenames + neg_filenames

x_train = []
x_test = []
for filename in filenames:
	count = 0
	with io.open(filename,'r', encoding='utf-8', errors='ignore') as f:
		line = f.readlines()
	for item in line:
		count += 1.0
		item = item.replace('<br />',' ')
		item = item.replace('\x96',' ')
		item = nltk.word_tokenize(item)
		item = [w.lower() for w in item]
		if count < 3732:
			x_train.append(item)
		else:
			x_test.append(item)
	print(count) # both have 5331 files

## number of tokens per review
no_of_tokens = []
for tokens in x_train:
	no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)

### word_to_id and id_to_word. associate an id to every unique token in the training dat
all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

## let's sort the indices by word frequency instead of random
x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
	for token in x:
		count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]
hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
for i in range(10):
	print(id_to_word[i],count[i])

## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

## save dictionary
np.save('../preprocessed_data/review_dictionary.npy',np.asarray(id_to_word))
## save training data to single text file
with io.open('../preprocessed_data/review_train.txt','w',encoding='utf-8') as f:
	for tokens in x_train_token_ids:
		for token in tokens:
			f.write("%i " % token)
		f.write("\n")
## save test data to single text file
with io.open('../preprocessed_data/review_test.txt','w',encoding='utf-8') as f:
	for tokens in x_test_token_ids:
		for token in tokens:
			f.write("%i " % token)
		f.write("\n")
