import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import time
import os
import sys
import io
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from BOW_model import BOW_model
from sklearn.metrics import accuracy_score

__RNN__ = 0
__SVM__ = 1
__naive_bayes__ = 1

__polarity__ = 0
__imdb__ = 1

#imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = 8000
x_train = []

if __polarity__:
	with io.open('../preprocessed_data/review_train.txt','r',encoding='utf-8') as f:
		lines = f.readlines()
if __imdb__:
	with io.open('../preprocessed_IMDB/review_train.txt','r',encoding='utf-8') as f:
		lines = f.readlines()

for line in lines:
	line = line.strip()
	line = line.split(' ')
	line = np.asarray(line,dtype=np.int)
	line[line>vocab_size] = 0
	x_train.append(line)

if __polarity__:
	x_train = x_train[0:3731 * 2]
	y_train = np.zeros((3731 * 2,))
	y_train[0:3731] = 1

if __imdb__:
	x_train = x_train[0:25000]
	y_train = np.zeros((25000,))
	y_train[0:12500] = 1

x_test = []
if __polarity__:
	with io.open('../preprocessed_data/review_test.txt','r',encoding='utf-8') as f:
		lines = f.readlines()
if __imdb__:
	with io.open('../preprocessed_IMDB/review_test.txt','r',encoding='utf-8') as f:
		lines = f.readlines()

for line in lines:
	line = line.strip()
	line = line.split(' ')
	line = np.asarray(line,dtype=np.int)
	line[line>vocab_size] = 0
	x_test.append(line)

if __polarity__:
	y_test = np.zeros((3200,))
	y_test[0:1600] = 1

if __imdb__:
	y_test = np.zeros((25000,))
	y_test[0:12500] = 1


# above we import the data, the below is the epochs running
if __RNN__ :
	vocab_size += 1
	model = BOW_model(vocab_size,500)
	#model.cuda()

	# opt = 'sgd'
	# LR = 0.01
	opt = 'adam'
	LR = 0.001
	if(opt=='adam'):
		optimizer = optim.Adam(model.parameters(), lr=LR)
	elif(opt=='sgd'):
		optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

	batch_size = 200
	no_of_epochs = 6
	L_Y_train = len(y_train)
	L_Y_test = len(y_test)
	model.train()
	train_loss = []
	train_accu = []
	test_accu = []
	test_loss = []

	for epoch in range(no_of_epochs):
		# training
		model.train()
		epoch_acc = 0.0
		epoch_loss = 0.0
		epoch_counter = 0
		time1 = time.time()
		I_permutation = np.random.permutation(L_Y_train)
		for i in range(0, L_Y_train, batch_size):
			x_input = [x_train[j] for j in I_permutation[i:i+batch_size]]
			y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]],dtype=np.float32)
			target = Variable(torch.FloatTensor(y_input))
			optimizer.zero_grad()
			loss, pred = model(x_input,target)
			loss.backward()
			optimizer.step() # update weights
			prediction = pred >= 0.0
			truth = target >= 0.5
			acc = prediction.eq(truth).sum().cpu().data.numpy()
			epoch_acc += acc
			epoch_loss += loss.data.item()
			epoch_counter += batch_size

		epoch_acc /= epoch_counter
		epoch_loss /= (epoch_counter/batch_size)
		train_loss.append(epoch_loss)
		train_accu.append(epoch_acc)
		print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()))
		# ## test
		model.eval()
		epoch_acc = 0.0
		epoch_loss = 0.0
		epoch_counter = 0
		time1 = time.time()
		I_permutation = np.random.permutation(L_Y_test)
		for i in range(0, L_Y_test, batch_size):
			x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
			y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]],dtype=np.float32)
			target = Variable(torch.FloatTensor(y_input))
			with torch.no_grad():
				loss, pred = model(x_input,target)
			prediction = pred >= 0.0
			truth = target >= 0.5
			acc = prediction.eq(truth).sum().cpu().data.numpy()
			epoch_acc += acc
			epoch_loss += loss.data.item()
			epoch_counter += batch_size

		epoch_acc /= epoch_counter
		epoch_loss /= (epoch_counter/batch_size)
		test_loss.append(epoch_loss)
		test_accu.append(epoch_acc)
		time2 = time.time()
		time_elapsed = time2 - time1
		print(" ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

	#plot
	#plot
	plt.figure()
	plt.plot(train_accu,label = 'train_accuracy')
	plt.plot(train_loss,label = 'train_loss')
	plt.plot(test_accu, label='test_accuracy')
	plt.plot(test_loss,label='test_loss')
	plt.legend()
	plt.show()

	torch.save(model,'BOW.model')
	data = [train_loss,train_accu,test_accu]
	data = np.asarray(data)
	np.save('data.npy',data)

# using other models provided by sklearn
train_vector = []
test_vector = []

for i in range(len(x_train)):
	lis = list(x_train[i])
	while len(lis) < 100:
		lis.append(0)
	if len(lis) > 100:
		lis = lis[0 : 100]
	arr = np.array(lis)
	train_vector.append(arr)

for i in range(len(x_test)):
	lis = list(x_test[i])
	while len(lis) < 100:
		lis.append(0)
	if len(lis) > 100:
		lis = lis[0 : 100]
	arr = np.array(lis)
	test_vector.append(arr)

if __SVM__:
	#print(train_vector)
	clf = svm.SVC(gamma='scale')
	clf.fit(train_vector, y_train)
	result = clf.predict(test_vector)
	print("accuracy using SVM model is: " + str(accuracy_score(result, y_test)))

if __naive_bayes__:
	gnb = GaussianNB()
	result = gnb.fit(train_vector, y_train).predict(test_vector)
	print("accuracy using naive bayes model is: " + str(accuracy_score(result, y_test)))