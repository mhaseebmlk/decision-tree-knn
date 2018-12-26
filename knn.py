"""
This file implements the KNN algorithm.
"""

import math
import numpy as np
import time
from k_folds_generator import KFoldsGenerator

class KNN:
	# make sure that k is an odd value since the number of features are odd?
	def __init__(self,k,distance_metric_code):

		self.k = k
		self.distance_metric_code=distance_metric_code
		self.training_data=None
		# Also add Minkowsky distance?
		self.distance_metric_names={
			0:'Euclidean Distance',
			1:'Manhattan Distance',
			2:'Cosine Similarity'
		}

	def train(self,training_data):

		if self.k > len(training_data):
			raise Exception('Can\'t have more neighbors than training examples! k = {} and number of training examples = {}'.format(self.k,len(training_data)))

		# Training is just maintaining the data array
		self.training_data=training_data

	def predict(self,test_data):
		"""
		Predicts the label of the test_data101_point based on the training data provided earlier
		"""

		if self.training_data==None: raise Exception('Must first train the model.')

		predictions=list()
		for x in test_data:
			predictions.append(self._predict(x))
			# print 'The label for {} is {}'.format(x,predictions[-1])
		return predictions

	def _predict(self,test_point):
		target_labels,distances=list(),list()
		for i in range(len(self.training_data)):
			# print 'training data point: {}, test data point: {}'.format(self.training_data[i],test_point)
			dist = self.compute_distance(self.training_data[i][:-1],test_point[:-1])
			distances.append((dist,i))

		# print 'distances before sorting: {}'.format(distances)
		distances.sort(key=lambda t: t[0])
		# print 'distances after sorting: {}'.format(distances)

		for i in range(self.k): target_labels.append(self.training_data[distances[i][1]][-1])
		# print 'The {}-nearest target labels are: {}'.format(self.k,target_labels)
		return self._get_majority_vote_label(target_labels)

	def get_accuracy(self,testing_set,predictions):
		"""
		Given the testing_set and the corresponding predictions for that testing set, compute the accuracy
		"""

		assert len(testing_set)==len(predictions), 'The number of predictions must be the same as the number of examples in the testing set!'
		assert len(testing_set)>0, 'The size of the testing_set for which to compute accuracy must be greater than 0!'

		num_correctly_predicted=0.0
		for i in range(len(testing_set)):
			real_label = testing_set[i][-1]
			predicted_label = predictions[i]
			if real_label==predicted_label: num_correctly_predicted+=1
		return num_correctly_predicted/len(testing_set)*100.0

	def compute_distance(self,point1,point2):
		assert len(point1)==len(point2), 'Lengths of the two points must be equal.'
		if self.distance_metric_code==0:
			return self.euclidean_distance(point1,point2)
		elif self.distance_metric_code==1:
			return self.manhattan_distance(point1,point2)
		elif self.distance_metric_code==2:
			return self.cosine_similarity(point1,point2)
		else: raise Exception('Invalid distance metric code {}'.format(self.distance_metric_code))

	def euclidean_distance(self,point1,point2):
		sum=0.0
		for i in range(len(point1)): 
			sum += ((float(point1[i])-float(point2[i]))**2)
		return math.sqrt(sum)

	def manhattan_distance(self,point1,point2):
		sum=0.0
		for i in range(len(point1)): 
			sum += abs(float(point1[i])-float(point2[i]))
		return sum

	def cosine_similarity(self,point1,point2):
		point1=map(lambda p: float(p),point1)
		point2=map(lambda p: float(p),point2)
		dot_product = np.dot((point1), (point2))
		norm_point1 = np.linalg.norm((point1))
		norm_point2 = np.linalg.norm((point2))
		return dot_product / (norm_point1 * norm_point2)

	def get_k(self): return self.k

	def get_distance_metric_name(self): return self.distance_metric_names[self.distance_metric_code]

	def _get_majority_vote_label(self,labels):
		counts_dict=dict()
		max_count=-1
		max_label=None
		for label in labels:
			if not label in counts_dict:
				counts_dict[label]=0
			counts_dict[label]+=1
			if counts_dict[label] > max_count: 
				max_label = label
				max_count = counts_dict[label]
		return max_label

	def get_feature_vector(self,data,feature_num):
		if feature_num>=len(data[0]):
			raise Exception('Feature number must be less than the length of the data rows.')
		return [x[feature_num] for x in data]

def print_stdout(stdout): print stdout

def tune_hyper_parameters(folds,folds_generator,plot_graph=False):
	"""

	This function will tune both the hyperparameters i.e. k and distance metric and return the best ones based on k-fold crossvalidaiton results

	If plot_graph is set to True then this function will also plot the graph of accuracies vs. different values of the hyperparameters in addition to returning the optimal values.

	"""

	num_distance_metrics=3
	assert num_distance_metrics <= 3 # KNN currently only supports 3 distance metrics

	k_values_range = range(1,74)

	distance_metric_codes=range(num_distance_metrics)
	k_values=filter(lambda x: x%2!=0, k_values_range)
	print k_values

	validation_set_percentage=20 

	# For every combination of K and distance metrics, for every iteration of k-folds, get the test and training sets, train the KNN, get the accuracy and F-scores for the training and validation, store their means in the scores list, and get the maximum_score one as the values of k and the distance metric
	parameter_scores=list()
	for k in k_values:
		for distance_metric_code in distance_metric_codes:

			knn=KNN(k,distance_metric_code)
			if k%10==9:
				print 'Running {}-fold cross-validation with k={} and distance metric: {}'.format(len(folds),knn.get_k(),knn.get_distance_metric_name())

			# Do k-fold cross-validation
			# K-fold's iterations with treating each fold as a different training set
			avg_validation_accuracy_sum=0.0
			# avg_test_accuracy_sum=0.0
			for i in range(0,len(folds)):

				test_set=folds[i]
				training_set=list()
				for j in range(0,i): training_set.extend(folds[j])
				for j in range(i+1,len(folds)): training_set.extend(folds[j])

				training_set,validation_set = folds_generator.get_training_and_validation_sets(training_set,validation_set_percentage)

				knn.train(training_set)
				# print 'training_set: {}\nvalidation_set: {}'.format(training_set,validation_set)

				predictions=knn.predict(validation_set)
				# print 'predictions: {}'.format(predictions)

				validation_set_accuracy=knn.get_accuracy(validation_set,predictions)
				avg_validation_accuracy_sum+=validation_set_accuracy

			avg_validation_accuracy=avg_validation_accuracy_sum/len(folds)
			results_tuple=(k,distance_metric_code,avg_validation_accuracy)
			parameter_scores.append(results_tuple)
	print parameter_scores
	parameter_scores.sort(key=lambda t: t[2],reverse=True)
	print parameter_scores
	print 'Optimal parameters are k={} and distance metric: {}'.format(parameter_scores[0][0],parameter_scores[0][1])

def main():

	filename='winequality-white.csv'
	num_folds=4
	folds_generator=KFoldsGenerator(filename,num_folds)
	folds=folds_generator.generate_folds(True)

	# ===================== HYPER PARAMETER TUNING =====================
	# tune_hyper_parameters(folds,folds_generator)
	# ==================================================================
	
	# ============ Training, Prediction, Performance Calculations ============
	start_time = time.time()

	k, distance_metric_code = 1,1
	knn = KNN(k,distance_metric_code)
	
	stdout='Hyper-parameters:\nK: {}\nDistance measure: {}\n'.format(knn.get_k(),knn.get_distance_metric_name())
	print_stdout(stdout)

	validation_set_percentage=20 

	avg_validation_accuracy_sum=0.0
	avg_validation_f1_sum=0.0

	avg_test_accuracy_sum=0.0
	avg_test_f1_sum=0.0

	avg_f1_score=0.0
	for i in range(0,len(folds)):
		stdout='Fold-{}:'.format(i+1)
		print_stdout(stdout)

		test_set=folds[i]
		training_set=list()
		for j in range(0,i): training_set.extend(folds[j])
		for j in range(i+1,len(folds)): training_set.extend(folds[j])
		training_set,validation_set = folds_generator.get_training_and_validation_sets(training_set,validation_set_percentage)
		knn.train(training_set)

		predictions=knn.predict(validation_set)
		validation_set_accuracy=knn.get_accuracy(validation_set,predictions)
		avg_validation_accuracy_sum+=validation_set_accuracy

		predictions=knn.predict(test_set)
		test_set_accuracy=knn.get_accuracy(test_set,predictions)
		avg_test_accuracy_sum+=test_set_accuracy

		stdout='Validation: F1 Score: {}, Accuracy: {}'.format(None,validation_set_accuracy)
		print_stdout(stdout)
		stdout='Test: F1 Score: {}, Accuracy: {}'.format(None,test_set_accuracy)
		print_stdout(stdout)
		stdout=''
		print_stdout(stdout)

	stdout='Average:\nValidation: F1 Score: {}, Accuracy: {}\nTest: F1 Score: {}, Accuracy: {}'.format(None,avg_validation_accuracy_sum/num_folds,None,avg_test_accuracy_sum/num_folds)
	print_stdout(stdout)
	stdout=''
	print_stdout(stdout)

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))
	# ==================================================================

if __name__=='__main__': main()