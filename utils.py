'''
Author: Gianfranco Lombardo
Project: WalkHubs2Vec
'''
import settings
settings.init()
from operator import itemgetter
import os
import deepwalk_functions
from gensim.models import Word2Vec
import time
import random
from skipgram import Skipgram
import networkx as nx
from scipy.linalg import orthogonal_procrustes
from math import sqrt
import numpy as np
from multiprocessing import Process
import pandas as pd

def read_edges_list_no_file(edges:list,graph):
	"""
	It takes a list of edges and a graph as input and adds the edges to the graph
	
	:param edges: a list of tuples, each tuple is an edge
	:param graph: the graph to which the edges will be added
	:return: The graph is being returned.
	"""
	
	assert edges!=None and graph!=None
	for edge in edges:
		
		node1, node2= edge
			#Do not consider auto-loop
		
		if (node1!=node2):
			graph.add_edge(node1, node2)
		else:
			print(f'autoloop con {node1}')
		
	return graph
def read_edges_list(source,graph,separator=","):
	"""
	It reads a file containing edges and adds them to a graph
	
	:param source: the file containing the edges
	:param graph: the graph object
	:param separator: The separator used in the file, defaults to   (optional)
	:return: The graph
	"""
	assert source!=None and graph!=None
	edges= pd.read_csv(source,sep=separator)
	for node1, node2 in zip(edges['Source'], edges['Target']):
		if(node1!=node2):
			#Do not consider auto-loop
			graph.add_edge(node1, node2)
		else: print(f'auto loop con {node1}')
	return graph
def export_nodes_set(path,nodeset):
	"""
	It takes a path and a set of nodes and writes the nodes to the file at the given path
	
	:param path: the path to the file you want to write to
	:param nodeset: a set of nodes
	"""
	with open(path,"w+") as output:
		for n in nodeset:
			output.write(n+"\n")
def export_graph(path,G):
	"""
	It takes a graph and a path, and writes the edges of the graph to a file at the given path.
	
	:param path: the path to the folder where the graph is stored
	:param G: the graph
	"""
	'''with open(path+"_nodes.csv","w+") as output:
		for n in G.nodes():
			output.write(n+"\n")'''
	with open(path+"_edges.csv","w+") as output:
		print(G.has_node(188483))
		for e in G.edges():
			print(e)
			if(e[0]==188483 or e[1]==188483):
				print("eccoci")
				input('eccoci')
			output.write(f"{e[0]} {e[1]}\n")

def traslation(X,normalizationAxis=0):
	"""
	> This function takes a matrix X and returns a matrix with the same dimensions as X, but with the
	mean of each column subtracted from each element in that column
	
	:param X: the data to be normalized
	:param normalizationAxis: 0 for normalizing each feature, 1 for normalizing each sample, defaults to
	0 (optional)
	:return: The result of the traslation and the traslation factor.
	"""
	if(len(X)>1):
		result = (X - np.mean(X, axis=normalizationAxis))
		traslationFactor =np.mean(X, axis=normalizationAxis)
	else:
		result = (X - np.zeros(settings.DIMENSION))
		traslationFactor = np.zeros(settings.DIMENSION)
	return result,traslationFactor

def scaling(X):
	"""
	It takes a matrix X and returns a matrix that is the same size as X, but with each element divided
	by the square root of the sum of the squares of all the elements in X
	
	:param X: the matrix to be scaled
	:return: The result of the scaling and the scaling factor.
	"""

	k = X.shape[0]

	scalingFactor = 0
	Xlist = X.flatten()
	for e in Xlist:
		scalingFactor+= float(e*e)
	scalingFactor= scalingFactor/k
	scalingFactor = sqrt(scalingFactor)
	result = X/scalingFactor
	return result,scalingFactor
	

def incremental_nodes_edges(name,I_nodeSet,G):
	"""
	It takes a node and a graph, and it creates a new graph with all the edges that contain the node
	
	:param name: the name of the thread
	:param I_nodeSet: the set of nodes to be processed
	:param G: the graph
	"""
	print ("Thread '" + name + "' avviato")
	zeros=0
	for node in I_nodeSet:
		i_graph = nx.Graph()
		if settings.DIRECTED:
			i_graph = nx.DiGraph()
		for e in G.edges():
			if node in e:
				i_graph.add_edge(e[0],e[1])
		export_graph(settings.INCREMENTAL_DIR+str(node),i_graph)

def extract_hub_component(G,threshold,verbose=False):
	"""
	It takes a graph G, a threshold (e.g. 30), and a verbose flag (default False) and returns a subgraph
	H of G that contains all the nodes of G that have a degree less than the threshold
	
	:param G: the graph
	:param threshold: the percentage of nodes to be removed from the graph
	:param verbose: if True, prints out the number of nodes in each component, the minimum and maximum
	degree of each component, and the percentage of the total graph that each component represents,
	defaults to False (optional)
	:return: A subgraph of the original graph G, where the subgraph is composed of the nodes in the A
	list.
	"""
	nd = sorted(G.degree(),key=itemgetter(1))
	B = []
	B_len = round(G.number_of_nodes()/100*(100-threshold)) # E.g IF hub split is 30 we calculate 100-30=70 for b_len
	
	for elem in nd:
		node_id = elem[0]
		node_degree = elem[1]
		if len(B) < B_len:
			B.append(node_id)
		else:
			break
	next_node_degree = G.degree(nd[B_len][0])
	# Each degree has to be entirely included in A or B. In this case, otherwise we delete the degree from B
	if G.degree(B[B_len-1]) == next_node_degree:
		B = [x for x in B if not G.degree(x) == next_node_degree]

	A = [x[0] for x in nd[len(B):]]
	
	if verbose:
		print('The '+str(100-threshold)+'% of ' + str(G.number_of_nodes()) + ' is about ' + str(B_len) + '\n')
		print('B length: ' + str(len(B)) + ' (' + str(round(100*len(B)/G.number_of_nodes(), 2)) + '%)')
		print('Min Degree in B: ' + str(G.degree(B[0])))
		print('Max Degree in B: ' + str(G.degree(B[len(B)-1])) + '\n')
		print('A length: ' + str(len(A))  + ' (' + str(round(100*len(A)/G.number_of_nodes(), 2)) + '%)')
		print('Min Degree in A: ' + str(G.degree(A[0])) )
		print('Max Degree in A: ' + str(G.degree(A[len(A)-1])) + '\n')
		
	H = G.subgraph(A)
	return H
def parallel_incremental_embedding(nodes_list,edges_lists,H,G,G_model,workers=2):
	"""
	It takes a list of nodes, a list of edges, a graph, a graph model, and a number of workers, and then
	it splits the nodes and edges into sets of nodes and edges, and then it creates a process for each
	set of nodes and edges, and then it starts all the processes, and then it waits for all the
	processes to finish
	
	:param nodes_list: list of nodes to be embedded
	:param edges_lists: a list of lists of edges. Each list of edges is a graph
	:param H: the graph we want to embed
	:param G: the graph we're trying to embed
	:param G_model: the model of the graph G
	:param workers: number of threads to use, defaults to 2 (optional)
	"""
	nodes_sets = [nodes_list[i::workers] for i in range(workers)]
	graph_sets = [edges_lists[i::workers] for i in range(workers)]
	processList = []
	t_c=0
	for ns in nodes_sets:
		p = Process(target=thread_incremental_embedding, args=("process-"+str(t_c),ns,graph_sets[nodes_sets.index(ns)],H,G,G_model))
		processList.append(p)
		t_c+=1
	for p in processList:
		p.start()
	for p in processList:
		p.join()
def thread_incremental_embedding(process_name,nodes_list,edges_lists,H,G,G_model):
	"""
	It takes a node, its edges, the hypergraph, the graph and the graph model and returns the embedding
	of the node
	
	:param process_name: the name of the process
	:param nodes_list: a list of nodes to be embedded
	:param edges_lists: a list of lists of edges. Each list of edges is the list of edges of a node
	:param H: the graph that we want to embed
	:param G: the graph
	:param G_model: the model of the graph G
	"""
	print(process_name+" started")
	for node in nodes_list:
		print(process_name+") processing node: "+node)
		emb =incremental_embedding(node,edges_lists[nodes_list.index(node)],H,G,G_model)
		
		content=node+" "
		for column in range(0,len(emb)):
				if column != len(emb) -1:
					content+=str(emb[column])+" "
				else:
					content+=str(emb[column])+"\n"

		settings.lck.acquire()
		with open(settings.INCREMENTAL_MODEL, 'a+') as out:
			out.write(content)
		settings.lck.release()

	print(process_name+" ended")


def incremental_embedding(node,edges_list,H,completeGraph,G_model):
	"""
	It takes a node, a list of edges, a graph H, a complete graph G, and a model G_model, and returns an
	embedding for the node
	
	:param node: the node to be added
	:param edges_list: the list of edges of the node to be added
	:param H: the graph that contains the hub nodes
	:param completeGraph: the graph that contains all the nodes
	:param G_model: the embedding of the complete graph
	:return: The embedding of the node to be added
	"""
	#TODO nodo 831 ha collegamento con nodo dello stesso anno
	G = completeGraph.copy()
	tmp = nx.Graph()
	tmp_nodes_added =[]
	if settings.DIRECTED:
		tmp = nx.DiGraph()
	tmp = read_edges_list_no_file(edges_list,tmp)
	H_plus_node = H.copy()
	H_init_edges_number = len(H_plus_node.edges())
	
	embeddable = False
	
	for e in tmp.edges():
		if((e[0]) in H.nodes() or (e[1]) in H.nodes()):
			#if node has a link with someone in Hubs
			H_plus_node.add_edge(e[0],e[1])
			embeddable = True

	if(H_init_edges_number == len(H_plus_node.edges())):
		#if node has NOT ANY link with someone in Hubs

		found = False
		it=0

		while(not found and it<len(tmp.edges())):
			e = list(tmp.edges())[it]
			for incident_vertex in e:
				if incident_vertex != node:
					if incident_vertex in G.nodes():
						#vertex linked with node is in G
						found = True
						#print(incident_vertex)
						G.add_edge(e[0],e[1])
						hub_node_found=False
						while not hub_node_found:
							for hubtmp in H.nodes():
								#h_node = random.choice(list(H.nodes()))
								exist = nx.has_path(G, source=node, target=hubtmp)
								if exist:
									break
							if not exist:
								#TODO creo un arco fittizio (sarà giusto?)
								H_plus_node.add_edge(incident_vertex,random.choice(list(H.nodes())))
								hub_node_found=True
								embeddable=True
							if(exist):
								sh_paths =nx.shortest_path(G, source=node, target=hubtmp, weight=None, method='dijkstra')
								#add this walk to H_plus_node
								for i in range(len(sh_paths)):
									if(i+1<len(sh_paths)):
										H_plus_node.add_edge(sh_paths[i],sh_paths[i+1])
								hub_node_found=True
								embeddable=True
			it+=1

	####### AT THIS POINT I'M GOOD WITH H 
	if(embeddable):
		model_i= None
		#print(H_plus_node.has_node(188483))
		if settings.BASE_ALGORITHM =="deepwalk":
			dfedges=nx.to_pandas_edgelist(H_plus_node)
			for n in H_plus_node.nodes():
				if H_plus_node.degree(n)==0:
					print(f'node {n} isolato nel grafo hub+nodo')
					dfedges.loc[len(dfedges.index)] = [n,n]
			with open(f"{settings.TMP}{node}_edges.csv","w+", newline='') as f:
				dfedges.to_csv(f,header=False, index=False,sep=' ')
			#nx.write_edgelist(H_plus_node, f"{settings.TMP}{node}_edges.csv", delimiter=' ',data=False)

			model_i=Deepwalk(f"{settings.TMP}{node}_edges.csv",settings.DIRECTED,settings.EMBEDDING_DIR,f"{node}_i",1,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS)
			
		elif settings.BASE_ALGORITHM =="node2vec":
			pass #TO DO
		
		assert model_i
		model_i_dict = extract_embedding_for_Hub_nodes(H_plus_node,model_i)
		#pre-processing for alignment
		
		e_i_raw = model_i[str(node)]
		
		i_neighboors = list(H_plus_node[node])
		H_plus_node.remove_node(node) #remove node to be incrematlly added
		H_plus_node_copy= H_plus_node.copy()
		H_model=extract_embedding_for_Hub_nodes(H_plus_node,G_model)
		
		A_embeddings = []
		B_embeddings = []
		neighboors=[]
		
		#takes neighboors of node in Hub graph and add also neighboors of these ones (second order neighboors)
		for n in i_neighboors:
			neighboors.append(n)
			second_order = list(H_plus_node[n])
			if(node in second_order):
				second_order.remove(node)
			neighboors= neighboors + second_order
		neighboors = set(neighboors)
		
		# Creating two lists of embeddings, A_embeddings and B_embeddings.
		#check if neighboors are in dictionary of embeddings of Hubs minus new node or in dictionary of hubs plus node
		for n in neighboors:
			for e in H_model:
				if e == n:
					A_embeddings.append(H_model[e])#e[1:settings.DIMENSION+1])
			for f in model_i_dict:
				if f == n:
					B_embeddings.append(model_i_dict[f])#f[1:settings.DIMENSION+1])
		
		
		A_embeddings,A_mean = traslation(A_embeddings)
		# If here we save embeddings A not scaled but only traslated
		A_embeddings, A_scale = scaling(A_embeddings)
		
		B_embeddings,B_mean = traslation(B_embeddings)
		B_embeddings, scalingFactor = scaling(B_embeddings)
		R, s = orthogonal_procrustes(B_embeddings,A_embeddings)

		e_i = e_i_raw - B_mean
		e_i = e_i/scalingFactor
		e_i = e_i.dot(R)
		#Rescale to A scale
		e_i = A_scale*e_i
		#Translate again into the original position
		e_i+=A_mean
		
		#Remove temporary files
		os.remove(f"{settings.TMP}{node}_edges.csv")
		return e_i
	
def Deepwalk(edges_file,edges_type,embedding_dir,embeddingName,emb_workers,window_size,representation_size,NUM_WALKS,LEN_WALKS, separator=' '):
	"""
	It takes an edge list file, a boolean value to indicate whether the graph is directed or not, the
	directory where the embedding will be saved, the name of the embedding, the number of workers, the
	window size, the representation size, the number of walks and the length of each walk. It then
	returns the trained model
	
	:param edges_file: the path to the file containing the edges of the graph
	:param edges_type: undirected or directed
	:param embedding_dir: The directory where the embeddings will be saved
	:param embeddingName: The name of the embedding. This will be used to name the embedding files
	:param emb_workers: number of workers to use for training the embedding
	:param window_size: The maximum distance between the current and predicted word within a sentence
	:param representation_size: The number of dimensions in the embedding
	:param NUM_WALKS: Number of walks per node
	:param LEN_WALKS: The length of each random walk
	:return: The model is being returned.
	"""
	start_time = time.time()
	G = deepwalk_functions.load_edgelist(edges_file, undirected= edges_type, separator=separator)
	walks_filebase = embedding_dir+embeddingName+".walks"
	walk_files = deepwalk_functions.write_walks_to_disk(G, walks_filebase, num_paths=NUM_WALKS,
                                         path_length=LEN_WALKS, alpha=0, rand=random.Random(0),
                                         )
	#print("walk_files is "+str(walk_files))
	vertex_counts = deepwalk_functions.count_textfiles(walk_files, emb_workers)

	#print("Training for "+embeddingName)
	walks_corpus = deepwalk_functions.WalksCorpus(walk_files)
	model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
					 size=representation_size,
					 window=window_size, min_count=0, trim_rule=None, workers=emb_workers)
	model.wv.save_word2vec_format(embedding_dir+"emb/"+embeddingName+".emb") #plain text
	model.save(embedding_dir+"bin/"+embeddingName+".bin")
	elapsed_time = time.time() - start_time
	#print("Elapsed time :"+str(elapsed_time))
	
	#Remove temporary files
	os.remove(walks_filebase+".0")
	return model
def extract_embedding_for_Hub_nodes(H,G_model):
	"""
	It takes a graph and a model and returns a dictionary of node embeddings for the nodes in the graph
	
	:param H: the hub graph
	:param G_model: the embedding model
	:return: A dictionary of the embeddings of the nodes in H.
	"""
	H_mod = {}
	for n in H.nodes():
		e_n=G_model[str(n)]
		H_mod[n]=e_n
	return H_mod
	