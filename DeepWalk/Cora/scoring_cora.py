import numpy as np
import pandas as pd
import glob
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csc_matrix

# It's a wrapper for OneVsRestClassifier that allows you to predict the top k labels for each sample
class TopKRanker(OneVsRestClassifier):
	def predict(self, X, top_k_list):
		assert X.shape[0] == len(top_k_list)
		probs = np.asarray(super(TopKRanker, self).predict_proba(X))
		all_labels = []
		for i, k in enumerate(top_k_list):
			probs_ = probs[i, :]
			labels = self.classes_[probs_.argsort()[-k:]].tolist()
			all_labels.append(labels)
		return all_labels
		
if __name__ == "__main__":

	embed_file = "results/cora_DySAT.emb"#"results/cora_ctdne.emb"#"results/cora_GN.emb"
	cora_filtered_nodes = "./results/nodes_gephi.csv"
	model = KeyedVectors.load_word2vec_format(embed_file, binary=False)
	#model_AA = KeyedVectors.load_word2vec_format("results/cora_inc.emb", binary=False)
	available_classes = {'Encryption_and_Compression': 0, 'Artificial_Intelligence': 1, 'Operating_Systems': 2, 'Programming': 3, 'Human_Computer_Interaction': 4, 'Information_Retrieval': 5, 'Networking': 6, 'Databases': 7, 'Hardware_and_Architecture': 8, 'Data_Structures__Algorithms_and_Theory': 9}
	features = []
	nodes = []
	classe = []
	y = []
	
	
	# Reading the file cora_filtered_nodes and appending the first and second column to the lists nodes
	# and classe.
	with open(cora_filtered_nodes) as input:
		input.readline()
		for row in input.readlines():
			fields = row.split(",")
			nodes.append(fields[0])
			classe.append(fields[1])

	# Creating a list of features and a list of labels.
	for n in nodes:
		if str(n) in model:
			features.append(model[str(n)])
			y.append(available_classes[classe[nodes.index(n)]])
	# Converting the list of features into a numpy array.
	features_matrix = np.asarray(features)

	# Creating a matrix of zeros with the number of rows equal to the number of labels in the list y and
	# the number of columns equal to the number of available classes.
	labels_matrix=np.zeros((len(y), len(available_classes)))
	for r in range(0,len(y)):
		row = r
		col = y[r] 
		labels_matrix[row][col] = 1

	labels_matrix = csc_matrix(labels_matrix)
	labels_count = labels_matrix.shape[1]
	mlb = MultiLabelBinarizer(range(labels_count))
# 2. Shuffle, to create train/test groups
	shuffles = []
	for x in range(10):
		shuffles.append(skshuffle(features_matrix, labels_matrix))
	
	# 3. to score each train/test group
	all_results = defaultdict(list)
	
	
	training_percents = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	for train_percent in training_percents:
		for shuf in shuffles:
	
			X, y = shuf
	
			training_size = int(train_percent * X.shape[0])
	
			X_train = X[:training_size, :]
			y_train_ = y[:training_size]
	
			# Creating a list of lists.
			y_train = [[] for x in range(y_train_.shape[0])]
	
	
			cy =	y_train_.tocoo()
			for i, j in zip(cy.row, cy.col):
					y_train[i].append(j)
	
			# Checking that the number of non-zero elements in the sparse matrix is equal to the number of
			# labels in the list of lists.
			assert sum(len(l) for l in y_train) == y_train_.nnz
	
			X_test = X[training_size:, :]
			y_test_ = y[training_size:]
	
			y_test = [[] for _ in range(y_test_.shape[0])]
	
			cy =	y_test_.tocoo()
			for i, j in zip(cy.row, cy.col):
					y_test[i].append(j)
	
			clf = TopKRanker(LogisticRegression())
			clf.fit(X_train, y_train_)
	
			# find out how many labels should be predicted
			# Creating a list of the number of labels for each sample in the test set.
			top_k_list = [len(l) for l in y_test]
			preds = clf.predict(X_test, top_k_list)
	
			results = {}
			averages = ["micro", "macro"]
			for average in averages:
					results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
	
			all_results[train_percent].append(results)
	
	print ('Results, using embeddings of dimensionality', X.shape[1])
	print ('-------------------')
	for train_percent in sorted(all_results.keys()):
		#print ('Train percent:', train_percent)
		#for index, result in enumerate(all_results[train_percent]):
			#print ('Shuffle #%d:	 ' % (index + 1), result)
		avg_score = defaultdict(float)
		for score_dict in all_results[train_percent]:
			for metric, score in iteritems(score_dict):
				avg_score[metric] += score
		for metric in avg_score:
			avg_score[metric] /= len(all_results[train_percent])
		#print ('Average score:', dict(avg_score))
		#print ('-------------------')
		print(str(train_percent)+","+str(avg_score))

'''np.savetxt("prova.txt",labels_matrix,fmt='%i')
print(labels_matrix)'''


'''
available_classes = set()
for e in y:
	available_classes.add(e)
print(available_classes)'''
