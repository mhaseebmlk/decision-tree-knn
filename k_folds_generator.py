"""
This file implements the class that will generate k folds of the data.
"""

import csv
import random
import math

class KFoldsGenerator:

	def __init__(self,dataset_path,num_folds):

		self.dataset_path=dataset_path
		self.num_folds=num_folds

		self.header=None

	def get_num_folds(): return self.num_folds
	def get_num_testing_folds(): return self.num_testing_folds
	def get_num_training_folds(): return self.num_training_folds

	def generate_folds(self,normalize_data):

		folds=list()

		header, data = None,None
		with open(self.dataset_path, 'rb') as csvfile:
			reader = csv.reader(csvfile,delimiter =';')
			# print reader
			header,data=next(reader),list(reader)
			# print header
			self.header=header
			# data=[r[0] for r in data]	

		random.Random(len(data)).shuffle(data)  

		if normalize_data==True: self.normalize_data(header,data)
		# print data

		folds_size=len(data)//self.num_folds
		remaining_examples_index=len(data)
		for i in range(self.num_folds):
			fold=data[i*folds_size:i*folds_size+folds_size]
			folds.append(fold)
			remaining_examples_index=(i*folds_size+folds_size)
			
		# If we are left with some data points and the number of remaining points is not that much, include them in the last fold
		if (float((len(data)-remaining_examples_index))/len(data) <= 0.05):
			folds[-1].extend(data[remaining_examples_index:])

		return folds

	def normalize_data(self,header,data):
		# print header
		# print data

		# get the min,max values of each feature
		features_min_max=list()
		for feature_num in range(len(header)-1):
			min_val,max_val=float('inf'),float('-inf')
			for x in data:
				if float(x[feature_num])<min_val: min_val = float(x[feature_num])
				if float(x[feature_num])>max_val: max_val = float(x[feature_num])
			features_min_max.append((min_val,max_val))
		# print features_min_max

		normalized_upper_bound=10.0
		for i in range(len(data)):
			for feature_num in range(len(header)-1):
				org_val = float(data[i][feature_num])
				data[i][feature_num] = self._normalize_value(org_val,features_min_max[feature_num][0],\
					features_min_max[feature_num][1])*normalized_upper_bound

	def _normalize_value(self,val,min_val,max_val):
		return (val-min_val)/(max_val-min_val)

	def get_training_and_validation_sets(self,training_set,validation_set_percentage):
		"""
		training_folds: An array of arrays holding the test folds.

		This function will go through the training folds and set aside validation_set_percentage amount of data as the validation data and the rest as the training data. Will return both of these data.
		"""
		validation_set_size=int(math.floor(len(training_set)*(validation_set_percentage/100.0)))
		validation_set=list()
		for _ in range(validation_set_size):
			validation_set.append(training_set.pop())
		return training_set,validation_set
