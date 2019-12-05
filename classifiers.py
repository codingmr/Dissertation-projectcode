import sys
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

import csv
import time

from sklearn.cross_validation import train_test_split

from sklearn import metrics

import numpy as np

class parseDNA:
	def __init__(self, chunkSize, numberChunks):

		# Initializes an object with some default values # interesting result at 3, 3
		## tries to ammend the default values to the arguments passed when script was run
		self.chunkSize = chunkSize
		self.numberChunks = numberChunks
		self.fileName = "Amino.fna"
		self.sampleSize = 2501
		self.lst = []

	def split_into_chunkgroups(self, chunkSize, numberChunks, indexPosition, data):
		# returns a tuple of the group of Chunks and the target
		
	    return [ data[j:j+chunkSize] for j in xrange(indexPosition, chunkSize*numberChunks	
			+indexPosition, chunkSize) ], data[j+chunkSize]

	def list_chunkgroups(self, chunkSize, numberChunks, data):
		# loops through each character calling split_into_chunksgroups
		# stays within the index bounds
		# returns a list of groups of Chunks and the target
		return [ self.split_into_chunkgroups(chunkSize, numberChunks, i, data) for i in 
				xrange(len(data)-(chunkSize*numberChunks+2)) ]

	def formatPrintAll(self):
		print '\n'.join('{}: {}'.format(*k) for k in enumerate(self.lst))

	def formatPrintOne(self, index):
		toString = ""

		for i in xrange(len(self.lst[index][0])):
			toString += "v" + str(i) + "=" + str(self.lst[index][0][i]) + " "
		toString += "with target == " + str(self.lst[index][1])

		print toString

	def parseFile(self):
		with open(self.fileName) as file:
			data = file.read()

			listData = []
			for word in data.split():
				if (len(listData) <= self.sampleSize):
					listData.append(word)
				else:
					break

			le = preprocessing.LabelEncoder()
			# Converting string labels into numbers.
			normalizedData = le.fit_transform(listData)
			data = normalizedData
			
			self.lst = self.list_chunkgroups(self.chunkSize, self.numberChunks, data)

	def MNB(self, x, y):

		X_train, X_test, y_train, y_test = train_test_split(x, y, 
							test_size=0.20,random_state=109) 

		#Create a Gaussian Classifier
		mnb = MultinomialNB()
		#Train the model using the training sets
		mnb.fit(X_train, y_train)

		#Predict the response for test dataset
		y_pred = mnb.predict(X_test)

		# Model Accuracy, how often is the classifier correct?
		print "Multi Bayes Accuracy:\t\t", metrics.accuracy_score(y_test, y_pred)

		with open('MNBResults2.csv', mode='a') as employee_file:
			employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', 
				quoting=csv.QUOTE_MINIMAL)

			employee_writer.writerow([self.chunkSize, self.numberChunks, 
				self.sampleSize, "Multinominal Naive Bayes", 
				metrics.precision_score(y_test, y_pred, average="macro"), 
				metrics.recall_score(y_test, y_pred, average="macro"), 
				metrics.accuracy_score(y_test, y_pred)])

	def GNB(self, x, y):
		X_train, X_test, y_train, y_test = train_test_split(x, y, 
							test_size=0.20,random_state=109) 

		#Create a Gaussian Classifier
		gnb = GaussianNB()
		#Train the model using the training sets
		gnb.fit(X_train, y_train)

		#Predict the response for test dataset
		y_pred = gnb.predict(X_test)

		# Model Accuracy, how often is the classifier correct?
		print "Bayes Accuracy:\t\t\t", metrics.accuracy_score(y_test, y_pred)

		with open('GNBResults2.csv', mode='a') as employee_file:
		    	employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', 
				quoting=csv.QUOTE_MINIMAL)

		   	employee_writer.writerow([self.chunkSize, self.numberChunks, 
				self.sampleSize, "Guassian Naive Bayes", metrics.precision_score
				(y_test, y_pred, average="macro"), metrics.recall_score(y_test, 
				y_pred, average="macro"), metrics.accuracy_score(y_test, y_pred)])

	def Kmeans(self, x, y):
		kmeans = KMeans(n_clusters=9, random_state=109).fit(x)

		#print kmeans.labels_
		print "K Means Accuracy:\t\t", metrics.accuracy_score(y, kmeans.labels_)

		with open('KMResults2.csv', mode='a') as employee_file:
		    	employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', 
				quoting=csv.QUOTE_MINIMAL)

		   	employee_writer.writerow([self.chunkSize, self.numberChunks, 
				self.sampleSize, "K Means", metrics.precision_score(y, 
				kmeans.labels_, average="macro"), metrics.recall_score(y, 
				kmeans.labels_, average="macro"), metrics.accuracy_score(y, 
				kmeans.labels_)])

	def RFC(self, x, y):
		X_train, X_test, y_train, y_test = train_test_split(x, y, 
							test_size=0.20,random_state=109) # 

		clf2 = RandomForestClassifier(n_estimators=14, random_state=109)
		clf2 = clf2.fit(X_train, y_train)
		y_pred = clf2.predict(X_test)

		# Model Accuracy, how often is the classifier correct?
		print "Random Forest Accuracy:\t\t", metrics.accuracy_score(y_test, y_pred)

		with open('RFCResults2.csv', mode='a') as employee_file:
		    	employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', 
				quoting=csv.QUOTE_MINIMAL)

		   	employee_writer.writerow([self.chunkSize, self.numberChunks, 
				self.sampleSize, "Random Forest Classifier", 
				metrics.precision_score(y_test, y_pred, average="macro"), 
				metrics.recall_score(y_test, y_pred, average="macro"), 
				metrics.accuracy_score(y_test, y_pred)])

	def classifierSelect(self):
		
		# Convert reshape arraylist
		labels=[]

		v0=[]
		vx=[]
		
		

		for i in xrange(len(self.lst)):
			vx.append(np.reshape(self.lst[i][0], self.chunkSize*self.numberChunks))
		for i in xrange(len(self.lst)):
			labels.append(self.lst[i][1])

		#print labels
		# Naive Bayes
		#print self.lst[1][0][0]
		#self.GNB(vx, labels)
		#self.MNB(vx, labels)
		# K Means
		#self.Kmeans(vx, labels)
		# Random Forest
		self.RFC(vx, labels)

	def setNumberChunks(self, numberChunks):
		self.numberChunks = numberChunks

	def setChunkSize(self, chunkSize):
		self.chunkSize = chunkSize

	def setSampleSize(self, sampleSize):
		self.sampleSize = sampleSize

with open('GNBResults3.csv', mode='w') as employee_file:
	employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	
	employee_writer.writerow(['ChunkSize', 'Chunks', 'SampleSize', 'Classifier', 'Precision', 'Recall', 'Accuracy'])

with open('RFCResults3.csv', mode='w') as employee_file:
	employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	
	employee_writer.writerow(['ChunkSize', 'Chunks', 'SampleSize', 'Classifier', 'Precision', 'Recall', 'Accuracy'])

with open('KMResults3.csv', mode='w') as employee_file:
	employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	
	employee_writer.writerow(['ChunkSize', 'Chunks', 'SampleSize', 'Classifier', 'Precision', 'Recall', 'Accuracy'])

with open('MNBResults3.csv', mode='w') as employee_file:
	employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	
	employee_writer.writerow(['ChunkSize', 'Chunks', 'SampleSize', 'Classifier', 'Precision', 'Recall', 'Accuracy'])

# Test 1
dnaObj = parseDNA(50, 50)

#for i in xrange(1, 30):
#	for j in xrange(1, 30):
#		dnaObj.setChunkSize(i)
#		dnaObj.setNumberChunks(j)
#		dnaObj.parseFile()
#		dnaObj.classifierSelect()


for i in xrange(1500, 13500, 500):
	dnaObj.setSampleSize(i)
	dnaObj.parseFile()
	dnaObj.classifierSelect()



print "All tests complete!"

