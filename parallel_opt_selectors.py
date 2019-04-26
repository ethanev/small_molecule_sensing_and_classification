import os
import itertools 
import argparse 
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform, euclidean
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', help='list of selector groups')
	parser.add_argument('-t', help='name to help id output file!')
	parser.add_argument('-n', help='number of selectors in use')
	args = parser.parse_args()
	selector_group = args.g
	sample_name = args.t
	sel_number = int(args.n)
	return selector_group,sample_name,sel_number

def get_classifier(classifier):
	if classifier == 'LR':
		return LogisticRegressionCV(cv=3, penalty='l1', solver='saga', n_jobs=-1)
	elif classifier == 'RF':
		return RandomForestClassifier(n_estimators=100, n_jobs=-1)
	elif classifier == 'KNN':
		return KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
	elif classifier == 'SVM': # using the default RBF kernel..
		return SVC() 
	elif classifier == 'GP':
		kernel = 1.0 * RBF(1.0)
		return GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
	else:
		print('You did not input a valid classifier, please use one of: KNN, LR or RF')
		return
    
def get_majority_vote(vote_count_dict):
	highest_count = 0
	key_highest = None
	shared_highest = []
	for k,v in vote_count_dict.items():
		if v > highest_count:
			highest_count = v
			key_highest = k
			shared_highest = [(k,v)]
		elif v == highest_count:
			shared_highest.append((k,v))
	if len(shared_highest) > 1:
		rand_int = random.randint(0,len(shared_highest)-1)
		key_highest = shared_highest[rand_int][0]
	return key_highest

def main():
	selector_groups, sample_name, sel_number = parse_arguments()
	selector_groups = ast.literal_eval(selector_groups)
	cheese = False
	liquor = False
	oil = True

	data = pickle.load(open('./oil_data.pkl','rb'))
	extracted_feat = pickle.load(open('./featurized_oil.pkl', 'rb'))

	classifier = 'RF'
	labels = np.array([[0,1,2]])
	labels = np.repeat(labels, 12, axis=1).flatten()

	if cheese:
		name_c = ['C1_{}'.format(i) for i in range(1,13)]
		name_m = ['M1_{}'.format(i) for i in range(1,13)]
		name_p = ['P1_{}'.format(i) for i in range(1,13)]
		names = name_c + name_m + name_p
	if liquor:
		name_r = ['R1_{}'.format(i) for i in range(1,13)]
		name_v = ['V1_{}'.format(i) for i in range(1,13)]
		name_w = ['W1_{}'.format(i) for i in range(1,13)]
		names = name_r + name_v + name_w   
	if oil:
		name_c = ['C1_{}'.format(i) for i in range(1,13)]
		name_o = ['O1_{}'.format(i) for i in range(1,13)]
		name_w = ['W1_{}'.format(i) for i in range(1,13)]
		names = name_r + name_v + name_w    
	extracted_feat = extracted_feat.loc[names] # this just organized the y direction...or first axis
	# to properly do selection, going to keep the blocks of selectors together
	blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23],
				[24,25,26,27],[28,29,30,31],[32,33,34,35]]
	block_combos = [(0,3,6),(0,3,7),(0,3,8),(0,4,6),(0,4,7),(0,4,8),(0,5,6),(0,5,7),(0,5,8),
					(1,3,6),(1,3,7),(1,3,8),(1,4,6),(1,4,7),(1,4,8),(1,5,6),(1,5,7),(1,5,8),
					(2,3,6),(2,3,7),(2,3,8),(2,4,6),(2,4,7),(2,4,8),(2,5,6),(2,5,7),(2,5,8)]

	column_names = list(extracted_feat)
	feature_accuracies = {}
	for selector_group in selector_groups:
		new_names = []
		for col in column_names:
			for num in selector_group:
				if 'S'+str(num)+'_' in col:
					new_names.append(col)
		minextracted = extracted_feat[new_names]
		accuracy = []
		for combo in block_combos: # looping over the different blocks 
			data_combined_sh, labels_sh = minextracted.values, labels
			mask_test = blocks[combo[0]]+blocks[combo[1]]+blocks[combo[2]]
			mask_train = [i for i in range(labels.shape[0]) if i not in mask_test]
			x_test =  data_combined_sh[mask_test,:]
			x_train =  data_combined_sh[mask_train,:]
			y_test = labels_sh[mask_test]
			y_train = labels_sh[mask_train]
			x_train, y_train = shuffle(x_train, y_train)
			x_train_list = np.split(x_train,sel_number, axis=1)
			x_test_list = np.split(x_test,sel_number, axis=1)
			pred_test_combined = []
			for j in range(sel_number):
				x_tr, x_ts = x_train_list[j], x_test_list[j]
				clf = get_classifier(classifier)
				clf.fit(x_tr, y_train)
				pred_test_combined.append(clf.predict(x_ts))
			pred_test_combined = np.concatenate(pred_test_combined)
			pred_test_combined = pred_test_combined.reshape((-1,y_test.shape[0]))
			combined_vote = []
			for i in range(y_test.shape[0]):
				unique, counts = np.unique(pred_test_combined[:,i], return_counts=True)
				counts = dict(zip(unique, counts))
				majority = get_majority_vote(counts)
				combined_vote.append(majority)
			combined_vote = np.asarray(combined_vote)
			accuracy.append(accuracy_score(y_test, combined_vote))
		accuracy = np.asarray(accuracy)
		feature_accuracies[str(selector_group)] = (accuracy.mean(), accuracy.std())

	#### ok now for the KNN for the given selectors groups. 
	start = 300
	end = 600
	classifier = 'KNN' 
	labels = np.array([[0,1,2]])
	labels = np.repeat(labels, 12, axis=1).flatten()
	selector_lists = selector_groups 
	KNN_results = {}
	for selector_list in selector_lists:
		accuracies = []
		for combo in block_combos:
			pred_test_combined = []
			for i in selector_list:
				selector = []
				for w in sorted(data.keys()):
					selector_data = data[w]['S'+str(i)]
					selector_data = selector_data.T.fillna(selector_data.mean(axis=1)).T 
					selector.append(selector_data)
				selector = pd.concat(selector, axis=1)
				selector = selector.values.T
				selector = selector[:, start:end]
				mask_test = blocks[combo[0]]+blocks[combo[1]]+blocks[combo[2]]
				mask_train = [i for i in range(labels.shape[0]) if i not in mask_test]
				x_test = selector[mask_test,:]
				x_train = selector[mask_train,:]
				y_test = labels[mask_test]
				y_train = labels[mask_train]
				clf = get_classifier(classifier)
				x_train, y_train = shuffle(x_train, y_train)
				clf.fit(x_train, y_train)
				pred_test_combined.append(clf.predict(x_test)) # might be better to work with probablilites... predict_proba
			pred_test_combined = np.concatenate(pred_test_combined)
			pred_test_combined = pred_test_combined.reshape((-1,y_test.shape[0]))
			combined_vote = []
			for i in range(y_test.shape[0]):
				unique, counts = np.unique(pred_test_combined[:,i], return_counts=True)
				counts = dict(zip(unique, counts))
				majority = get_majority_vote(counts)
				combined_vote.append(majority)
			combined_vote = np.asarray(combined_vote)
			accuracies.append(accuracy_score(y_test, combined_vote))
		accuracies = np.asarray(accuracies)
		KNN_results[str(selector_list)] = (accuracies.mean(), accuracies.std())

	#### GP classification
#	classifier = 'GP' 
#	selector_lists = selector_groups 
#	GP_results = {}
#	for selector_list in selector_lists:
#		accuracies = []
#		for combo in block_combos:
#			pred_test_combined = []
#			for i in selector_list:
#				selector = []
#				for w in sorted(data.keys()):
#					selector_data = data[w]['S'+str(i)]
#					selector_data = selector_data.T.fillna(selector_data.mean(axis=1)).T 
#					selector.append(selector_data)
#				selector = pd.concat(selector, axis=1)
#				selector = selector.values.T
#				selector = selector[:, start:end]
#				mask_test = blocks[combo[0]]+blocks[combo[1]]+blocks[combo[2]]
#				mask_train = [i for i in range(labels.shape[0]) if i not in mask_test]
#				x_test = selector[mask_test,:]
#				x_train = selector[mask_train,:]
#				y_test = labels[mask_test]
#				y_train = labels[mask_train]
#				clf = get_classifier(classifier)
#				x_train, y_train = shuffle(x_train, y_train)
#				clf.fit(x_train, y_train)
#				pred_test_combined.append(clf.predict(x_test)) # might be better to work with probablilites... predict_proba
#			pred_test_combined = np.concatenate(pred_test_combined)
#			pred_test_combined = pred_test_combined.reshape((-1,y_test.shape[0]))
#			combined_vote = []
#			for i in range(y_test.shape[0]):
#				unique, counts = np.unique(pred_test_combined[:,i], return_counts=True)
#				counts = dict(zip(unique, counts))
#				majority = get_majority_vote(counts)
#				combined_vote.append(majority)
#			combined_vote = np.asarray(combined_vote)
#			accuracies.append(accuracy_score(y_test, combined_vote))
#		accuracies = np.asarray(accuracies)
#		GP_results[str(selector_list)] = (accuracies.mean(), accuracies.std())

	combined_data = {'KNN': KNN_results, 'features': feature_accuracies}
	print(combined_data)
	pickle.dump(combined_data,open('./combined_data_{}.pkl'.format(sample_name), 'wb'))

if __name__ == "__main__":
	main()
