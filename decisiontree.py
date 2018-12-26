"""
This file implements the ID3 algorithm.
"""

import math
import numpy as np
import time
from k_folds_generator import KFoldsGenerator

class DecisionTree:
	def __init__(self,max_depth):
		self.max_depth=max_depth
		self.tree=None
		self.tree_depth=0

	def train(self,training_data):
		# print 'Train the decision tree on data: {}'.format(training_data,len(training_data))

		# dummy_data=[
		# 	[0.7,0.27,8.8,4],
		# 	[6.3,0.3,9.5,4],
		# 	[8.1,0.28,10.1,4]
		# 	# [3.1,1.28,5.1,7]
		# ]

		self.tree=self.id3_v1(training_data,0)
		# print self.tree

	def id3_v1(self,data,depth=0):
		# if depth>self.max_depth: 
		# 	print 'Max depth reached! depth={} and max_depth={}'.format(depth,self.max_depth)
		# 	return

		if self.get_entropy(data) == 0 or depth>=self.max_depth:
			prediction_label=len(data[0])-1
			# return the label value 
			prediction_label_vec=self.get_feature_vector(data,prediction_label)

			if depth>self.tree_depth: self.tree_depth=depth

			return prediction_label_vec[0]

		best_feature=self.get_best_feature(data)
		# print 'best_feature',best_feature
		split_threshold=self.get_feature_split_thresh(data,best_feature)
		# print 'split_threshold',split_threshold,'split attribute',best_feature
		
		tree={best_feature:{'threshold':None,'left':None,'right':None}}
		tree[best_feature]['threshold']=split_threshold
		if depth>self.tree_depth: self.tree_depth=depth
		# print 'tree:',tree

		leq_data_subset=self.filter_data(data,best_feature,split_threshold,'<=')
		gt_data_subset=self.filter_data(data,best_feature,split_threshold,'>')
		# print 'leq_data_subset',leq_data_subset
		# print 'gt_data_subset',gt_data_subset

		# remove the best_feature attribute from the data now as well

		if len(leq_data_subset)==0:
			tree[best_feature]['left']=self.get_majority_target_label_val(data)
		else:
			tree[best_feature]['left']=self.id3_v1(leq_data_subset,depth+1)
		if len(gt_data_subset)==0:
			tree[best_feature]['right']=self.get_majority_target_label_val(data)
		else:
			tree[best_feature]['right']=self.id3_v1(gt_data_subset,depth+1)

		return tree

	def get_majority_target_label_val(self,data):
		assert len(data)>0, 'The data must contain at least one examples!'
		assert len(data[0])>2, 'The data must contain atleast 2 columns!'
		label_counts=dict()
		prediction_label=len(data[0])-1
		for x in data:
			if not x[prediction_label] in label_counts:
				label_counts[x[prediction_label]]=0
			label_counts[x[prediction_label]]+=0
		# get the maximum count label
		max_count,majority_label=-1,None
		for key in label_counts:
			if label_counts[key]>max_count:
				max_count=label_counts[key]
				majority_label=key
		# print 'The majority label in the data {} is {}'.format(data,majority_label)
		return majority_label

	def get_best_feature(self,data):
		"""
		Return the index in the dataset of the best feature (not the actual name of the best feature)
		"""
		feature_scores=list()
		for feature in range(len(data[0])-1):
			feature_scores.append( 
				(feature, self.get_information_gain(data,feature)) )
		best_feature=sorted(feature_scores,key=lambda t:t[1],reverse=True)[0][0]
		return best_feature

	def get_feature_split_thresh(self,data,feature):
		"""
		Gets the feature matrix consisting of this feature and the label columns, sorts them, and calculates the best value for the split threshold for this feature.

		This function is required since the feature values are continuous and not categorical.
		"""
		assert len(data[0])>2, 'The data must contain atleast 2 columns to find the best split threshold value.'

		target_label=len(data[0])-1
		feature_matrix=self.get_feature_matrix(data,[feature,target_label],True)
		feature_matrix.sort(key=lambda l:l[0])

		if len(feature_matrix)==1: return feature_matrix[0][0]
		if len(feature_matrix)==0: raise Exception('Feature_matrix does not exist.')
		
		# check distances between adj points
		max_dist=float('-inf')
		threshold_range=[feature_matrix[0][0],feature_matrix[len(feature_matrix)-1][0]]
		for i in range(len(feature_matrix)-1):
			curr_point=feature_matrix[i]
			adj_point=feature_matrix[i+1]
			if curr_point[1] != adj_point[1]:
				dist=abs(adj_point[0]-curr_point[0])
				# print '{}, and {} have diff labels and dist is {}'.format(curr_point,adj_point,dist)
				if dist>max_dist:
					max_dist=dist
					threshold_range[0]=curr_point[0]
					threshold_range[1]=adj_point[0]
		avg_val=reduce(lambda x, y: x + y, threshold_range) / len(threshold_range)
		# print 'threshold info:',max_dist,threshold_range,avg_val,threshold_range
		return avg_val

	def get_entropy(self,data):
		"""
		Will calculate and return the entropy of the data_set. The prediction label is the last column (quality)

		The dataset can be the whole dataset or a subset of the dataset but must contain all the feature columns
		"""

		assert len(data)>0, 'The data must contain at least one examples!'
		assert len(data[0])>2, 'The data must contain atleast 2 columns to find the entropy.'
	
		prediction_label=len(data[0])-1
		feature_vec = self.get_feature_vector(data,prediction_label)
		els,cnts = np.unique(feature_vec,return_counts = True)
		sum_=0.0
		for i in range(len(els)):
			probability=float(cnts[i])/np.sum(cnts)
			sum_ += -(probability*math.log(probability,2))
		return sum_

	def get_information_gain(self,data,feature_num):
		"""
		Given the data and the feature num (column number in the data), get the information gain for this feature. the target label column is alwys the last column
		"""
		assert len(data)>0, 'The data must contain more than one examples!'
		assert 0 <= feature_num < (len(data[0])-1), 'The feature number must be in the correct range.'

		feature_vec=self.get_feature_vector(data,feature_num)
		feature_uniq_vals=set(feature_vec)
		data_entropy=self.get_entropy(data)
		sum_=0.0
		for uniq_val in feature_uniq_vals:
			data_subset = self.filter_data(data,feature_num,uniq_val,'==')
			entropy_subset = self.get_entropy(data_subset)
			sum_ += ((len(data_subset)+0.0)/len(data))*(entropy_subset)
			# print data_subset,entropy_subset
		return data_entropy-sum_

	def filter_data(self,data,feature_num,feature_val,op):
		"""
		Given the data, it will traverse all the rows and compare the value in the feature_num column for that row. If that value satisfies the op then that example is added to the subset data.
		op is just a string and can take values in ['==','<','<=','>','>=','!=']
		"""
		assert len(data)>0, 'The data must contain more than one examples!'
		assert 0 <= feature_num < (len(data[0])-1), 'The feature number must be in the correct range.'
		
		data_sbst=None
		if op=='==':
			data_sbst=filter(lambda x:float(x[feature_num])==float(feature_val),data)
		elif op=='<':
			data_sbst=filter(lambda x:float(x[feature_num])<float(feature_val),data)
		elif op=='<=':
			data_sbst=filter(lambda x:float(x[feature_num])<=float(feature_val),data)
		elif op=='>':
			data_sbst=filter(lambda x:float(x[feature_num])>float(feature_val),data)
		elif op=='>=':
			data_sbst=filter(lambda x:float(x[feature_num])>=float(feature_val),data)
		elif op=='!=':
			data_sbst=filter(lambda x:float(x[feature_num])!=float(feature_val),data)
		else:
			raise Exception('Unsupported op provided: {}'.format(op))
		return data_sbst

	def predict(self,test_data):
		if not self.tree: raise Exception('Must first learn the Decision Tree')

		# d=self.tree
		# for i in range(10):
		# 	print type(d),d
		# 	d=d[d.keys()[0]]

		predictions=list()
		for x in test_data:
			predictions.append(self._predict(self.tree,x))
			# print 'The label for {} is {}'.format(x,predictions[-1])
		return predictions

	def _predict(self,tree_node,test_point):
		# print 'Predict the label for the point {}'.format(test_point)
		if isinstance(tree_node,str) or isinstance(tree_node,int) or isinstance(tree_node,float):
			# print '\tpredicted label for {} is {}'.format(test_point,tree_node)
			return tree_node
		if isinstance(tree_node,dict):
			node_feature=tree_node.keys()[0]
			values=tree_node[node_feature]
			feature_threshold=values['threshold']
			if float(test_point[node_feature]) <= float(feature_threshold):

				# print 'feature num {}, threshold {}, new point value {} new point {} -> less/equal -> checking left subtree'.format(node_feature,feature_threshold,test_point[node_feature],test_point)

				return self._predict(values['left'],test_point)

			# print 'feature num {}, threshold {}, new point value {} new point {} -> greater -> checking right subtree'.format(node_feature,feature_threshold,test_point[node_feature],test_point)

			return self._predict(values['right'],test_point)
		raise Exception('Encountered a tree node of type {} where it was expected to be either str,int,float or dict'.format(type(tree_node)))

	def get_max_depth(self): return self.max_depth

	def get_feature_vector(self,data,feature_num):
		assert len(data)>0, 'The data must contain more than one examples!'
		assert 0 <= feature_num < (len(data[0])), 'The feature number must be in the correct range.'
		return [x[feature_num] for x in data]

	def get_feature_matrix(self,data,feature_nums,convert_to_float=False):
		assert len(data)>0, 'The data must contain more than one examples!'
		assert len(feature_nums)<=len(data[0]), 'Number for features to get must be less than or equal to total number of features in the data'

		feature_vec=[]
		for x in data:
			vec=[]
			for feature_num in feature_nums:
				assert 0 <= feature_num < (len(data[0])), 'The feature number must be in the correct range.'
				if convert_to_float: vec.append(float(x[feature_num]))
				else: vec.append(x[feature_num])
			feature_vec.append(vec)
		return feature_vec

	def get_accuracy(self,testing_set,predictions):
		assert len(testing_set)==len(predictions), 'The number of predictions must be the same as the number of examples in the testing set!'
		assert len(testing_set)>0, 'The size of the testing_set for which to compute accuracy must be greater than 0!'
		num_correctly_predicted=0.0
		for i in range(len(testing_set)):
			real_label = testing_set[i][-1]
			predicted_label = predictions[i]
			if real_label==predicted_label: num_correctly_predicted+=1
		return num_correctly_predicted/len(testing_set)*100.0

def print_stdout(stdout): print stdout

def tune_hyper_parameters(folds,folds_generator,plot_graph=False):
	"""

	This function will tune the hyperparameter i.e. the tree depth and return the best ones based on k-fold crossvalidaiton results

	If plot_graph is set to True then this function will also plot the graph of accuracies vs. different values of the hyperparameters in addition to returning the optimal values.

	"""

	print 'Tuning hyperparameters...'

	depth_values_range = range(35,60)

	validation_set_percentage=20 

	parameter_scores=list()
	for d in depth_values_range:
		dt = DecisionTree(d)

		if d%10==9:
			print 'Running {}-fold cross-validation with max_depth={}'.format(len(folds),d)

		learnt_tree_depth=None
		avg_validation_accuracy_sum=0.0	
		for i in range(0,len(folds)):

			test_set=folds[i]
			training_set=list()
			for j in range(0,i): training_set.extend(folds[j])
			for j in range(i+1,len(folds)): training_set.extend(folds[j])

			training_set,validation_set = folds_generator.get_training_and_validation_sets(training_set,validation_set_percentage)

			dt.train(training_set)
			learnt_tree_depth = dt.tree_depth #tree's actual depth mght be smaller than the max depth
			predictions=dt.predict(validation_set)
			validation_set_accuracy=dt.get_accuracy(validation_set,predictions)
			avg_validation_accuracy_sum+=validation_set_accuracy

		avg_validation_accuracy=avg_validation_accuracy_sum/len(folds)
		results_tuple=(learnt_tree_depth,avg_validation_accuracy)
		parameter_scores.append(results_tuple)

	print 'Unsorted scores:',parameter_scores
	parameter_scores.sort(key=lambda t: t[1],reverse=True)
	print 'Sorted scores:',parameter_scores
	print 'Optimal max_depth is {} which gives {} percent accuracy on validation set'.format(parameter_scores[0][0],parameter_scores[0][1])

	return parameter_scores[0][0]

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

	max_depth = 44
	dt = DecisionTree(max_depth)

	stdout='Hyper-parameters:\nMax-Depth: {}\n'.format(dt.get_max_depth())
	print_stdout(stdout)

	validation_set_percentage=20 

	avg_validation_accuracy_sum=0.0
	avg_validation_f1_sum=0.0

	avg_test_accuracy_sum=0.0
	avg_test_f1_sum=0.0

	avg_training_accuracy_sum=0.0
	avg_training_f1_sum=0.0

	avg_f1_score=0.0
	for i in range(0,len(folds)):
		stdout='Fold-{}:'.format(i+1)
		print_stdout(stdout)

		test_set=folds[i]
		training_set=list()
		for j in range(0,i): training_set.extend(folds[j])
		for j in range(i+1,len(folds)): training_set.extend(folds[j])
		training_set,validation_set = folds_generator.get_training_and_validation_sets(training_set,validation_set_percentage)
		dt.train(training_set)

		predictions=dt.predict(validation_set)
		validation_set_accuracy=dt.get_accuracy(validation_set,predictions)
		avg_validation_accuracy_sum+=validation_set_accuracy

		predictions=dt.predict(test_set)
		test_set_accuracy=dt.get_accuracy(test_set,predictions)
		avg_test_accuracy_sum+=test_set_accuracy

		predictions=dt.predict(training_set)
		training_set_accuracy=dt.get_accuracy(training_set,predictions)
		avg_training_accuracy_sum+=training_set_accuracy

		stdout='Training: F1 Score: {}, Accuracy: {}'.format(None,training_set_accuracy)
		print_stdout(stdout)
		stdout='Validation: F1 Score: {}, Accuracy: {}'.format(None,validation_set_accuracy)
		print_stdout(stdout)
		stdout='Test: F1 Score: {}, Accuracy: {}'.format(None,test_set_accuracy)
		print_stdout(stdout)
		stdout=''
		print_stdout(stdout)

	stdout='Average:\nTraining: F1 Score: {}, Accuracy: {}\nValidation: F1 Score: {}, Accuracy: {}\nTest: F1 Score: {}, Accuracy: {}'.format(None,avg_training_accuracy_sum/num_folds,None,avg_validation_accuracy_sum/num_folds,None,avg_test_accuracy_sum/num_folds)
	print_stdout(stdout)
	stdout=''
	print_stdout(stdout)

	end_time = time.time()
	print("Elapsed time was %g seconds" % (end_time - start_time))
	# ==================================================================


if __name__=='__main__': main()