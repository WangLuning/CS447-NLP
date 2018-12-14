import numpy as np
import os
import nltk
import itertools
import io
import matplotlib.pyplot as plt

## create directory to store preprocessed data
if(not os.path.isdir('../preprocessed_IMDB')):
	os.mkdir('../preprocessed_IMDB')

## get all of the training reviews (including unlabeled reviews)
train_directory = '../aclImdb/train/'
pos_filenames = os.listdir(train_directory + 'pos/')
neg_filenames = os.listdir(train_directory + 'neg/')
unsup_filenames = os.listdir(train_directory + 'unsup/')
pos_filenames = [train_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [train_directory+'neg/'+filename for filename in neg_filenames]

filenames = pos_filenames + neg_filenames
count = 0
x_train = []
for filename in filenames:
	with io.open(filename,'r',encoding='utf-8') as f:
		line = f.readlines()[0]
	line = line.replace('<br />',' ')
	line = line.replace('\x96',' ')
	line = nltk.word_tokenize(line)
	line = [w.lower() for w in line]
	x_train.append(line)
	count += 1
	#print(count)

## get all of the test reviews
test_directory = '../aclImdb/test/'
pos_filenames = os.listdir(test_directory + 'pos/')
neg_filenames = os.listdir(test_directory + 'neg/')
pos_filenames = [test_directory+'pos/'+filename for filename in pos_filenames]
neg_filenames = [test_directory+'neg/'+filename for filename in neg_filenames]
filenames = pos_filenames+neg_filenames

count = 0
x_test = []
for filename in filenames:
	with io.open(filename,'r',encoding='utf-8') as f:
		line = f.readlines()[0]
	line = line.replace('<br />',' ')
	line = line.replace('\x96',' ')
	line = nltk.word_tokenize(line)
	line = [w.lower() for w in line]
	x_test.append(line)
	count += 1
	#print(count)

## number of tokens per review
no_of_tokens = []
for tokens in x_train:
	no_of_tokens.append(len(tokens))
no_of_tokens = np.asarray(no_of_tokens)
print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ', np.max(no_of_tokens))

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

hist, bins = np.histogram(count, bins=[1, 10, 100, 1000, 10000, 100000])
bin = ['1 - 10', '10 - 100', '100 - 1000', '1000 - 10000', '10000 - 100000']

pos = np.arange(len(bin))
width = 0.5     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos)
ax.set_xticklabels(bin)
plt.xlabel('frequency')
plt.ylabel('count')
plt.title('Frequency distribution of words')
plt.bar(pos, hist, width, color='b')
plt.show()

for i in range(10):
	print(id_to_word[i],count[i])

## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}
## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

## save dictionary
np.save('../preprocessed_IMDB/review_dictionary.npy',np.asarray(id_to_word))
## save training data to single text file
with io.open('../preprocessed_IMDB/review_train.txt','w',encoding='utf-8') as f:
	for tokens in x_train_token_ids:
		for token in tokens:
			f.write("%i " % token)
		f.write("\n")
## save test data to single text file
with io.open('../preprocessed_IMDB/review_test.txt','w',encoding='utf-8') as f:
	for tokens in x_test_token_ids:
		for token in tokens:
			f.write("%i " % token)
		f.write("\n")

