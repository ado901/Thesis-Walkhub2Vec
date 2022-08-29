'''
Author: Gianfranco Lombardo
Project: WalkHubs2Vec
'''
from typing import Tuple, Union
import settings
settings.init()
import deepwalk
from operator import itemgetter
import os
from gensim.models import Word2Vec, KeyedVectors
import time
import random
import networkx as nx
if settings.BASE_ALGORITHM == 'node2vec':
	import torch_geometric.nn as nn
	import torch
	from torch_geometric.data import Data
	import torch_geometric
from scipy.linalg import orthogonal_procrustes
from math import sqrt
import numpy as np
import multiprocess as mp
import pandas as pd
import time
def read_edges_list_no_file(edges:list,graph: Union[nx.DiGraph,nx.Graph])-> Union[nx.DiGraph,nx.Graph]:
	"""
	It takes a list of edges and a graph as input and adds the edges to the graph
	
	:param edges: a list of tuples, each tuple is an edge
	:param graph: the graph to which the edges will be added
	:return: The graph is being returned.
	"""
	
	assert edges!=None and graph!=None
	for edge in edges:
		
		node1, node2= edge
			#Do not consider auto-loop TODO
		graph.add_edge(node1, node2)
			
		
	return graph
""" def read_edges_list(source,graph,separator=","):
	assert source!=None and graph!=None
	edges= pd.read_csv(source,sep=separator)
	for node1, node2 in zip(edges['Source'], edges['Target']):
		if(node1!=node2):
			#Do not consider auto-loop TODO
			graph.add_edge(node1, node2)
		else: 
			print(f'auto loop con {node1}')
			graph.add_edge(node1, node2)
	return graph """

def traslation(X:list[list],normalizationAxis=0):
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

def extract_hub_component(G: Union[nx.DiGraph,nx.Graph],threshold=20,verbose=False) -> Union[nx.Graph,nx.DiGraph]:
	"""
	It takes a graph and a threshold, and returns a subgraph of the original graph that contains hubs under a threshold 
	
	:param G: The graph you want to split
	:type G: Union[nx.DiGraph,nx.Graph]
	:param threshold: the percentage of nodes to be removed from the graph, defaults to 20 (optional)
	:param verbose: If True, prints out the number of nodes in each set, the minimum and maximum degree
	of each set, and the percentage of nodes in each set, defaults to False (optional)
	:return: A subgraph of the original graph G, containing only the nodes in A.
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

def parallel_incremental_embedding(nodes_list:list[int],edges_lists:list[list[int]],H: Union[nx.DiGraph,nx.Graph],G:Union[nx.DiGraph,nx.Graph],G_model:KeyedVectors,workers=2):
	'''It takes a list of nodes, a list of edges, a graph, a model, and a number of workers, and then it
	splits the list of nodes into a number of sublists equal to the number of workers, and then it
	splits the list of edges into a number of sublists equal to the number of workers, and then it
	creates a number of processes equal to the number of workers, and then it starts the processes, and
	then it waits for the processes to finish
	
	Parameters
	----------
	nodes_list : list[int]
		list of nodes in the graph
	edges_lists : list[list[int]]
		list of lists of edges, where each list of edges is a list of edges for a given node
	H : Union[nx.DiGraph,nx.Graph]
		Union[nx.DiGraph,nx.Graph]
	G : Union[nx.DiGraph,nx.Graph]
		the graph to be embedded
	G_model : KeyedVectors
		the embedding model
	workers, optional
		number of processes to fork
	
	'''
	#in case of multiprocessing in spawn mode: Digraph not supported for pickle mode
	G_dict=nx.to_dict_of_dicts(G)
	H_dict=nx.to_dict_of_dicts(H)
	#pool = ProcessPool(nodes=workers)
	nodes_sets = [nodes_list[i::workers] for i in range(workers)]
	graph_sets = [edges_lists[i::workers] for i in range(workers)]
	if os.path.exists(f'{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.INCREMENTAL_MODEL}_{settings.BASE_ALGORITHM}_{settings.YEAR_START+1}.csv'):
		os.remove(f'{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.INCREMENTAL_MODEL}_{settings.BASE_ALGORITHM}_{settings.YEAR_START+1}.csv')
	if os.path.exists(f'{settings.DIRECTORY}tmp/nodetoeliminate.csv'):
		os.remove(f'{settings.DIRECTORY}tmp/nodetoeliminate.csv')
	processList = []
	t_c=0
	for ns in nodes_sets:
		p = mp.Process(target=thread_incremental_embedding, args=("process-"+str(t_c),ns,graph_sets[nodes_sets.index(ns)],H,G,G_model,))
		processList.append(p)
		t_c+=1
	for p in processList:
		p.start()
	for p in processList:
		p.join()

def thread_incremental_embedding(process_name:str,nodes_list:list[int],edges_lists:list[list],H:Union[nx.DiGraph,nx.Graph],G:Union[nx.DiGraph,nx.Graph],G_model:KeyedVectors):
	'''It takes a list of nodes, a list of edges, a graph H, a graph G, and a word2vec model G_model, and
	for each node in the list of nodes, it computes the incremental embedding of that node, and then
	writes the embeddings to a file
	
	Parameters
	----------
	process_name : str
		str
	nodes_list : list[int]
		list of nodes to be embedded
	edges_lists : list[list]
		list of lists of edges. Each list of edges is the list of edges that were added to the graph in the
	current year.
	H : Union[nx.DiGraph,nx.Graph]
		the graph that we want to embed
	G : Union[nx.DiGraph,nx.Graph]
		the graph to be embedded
	G_model : KeyedVectors
		the embedding model of the base graph
	
	'''
	#in case of spawn mode: Digraph not supported for pickle
	""" G=nx.from_dict_of_dicts(G_dict,create_using=nx.DiGraph())
	H=nx.from_dict_of_dicts(H_dict,create_using=nx.DiGraph()) """
	print(f"{process_name} started ")
	total= len(nodes_list)
	list_of_emb=[]
	count=0
	embs={}
	for node in nodes_list:
		print(F'{process_name} processing node: {node}')
		start_time= time.process_time() 
		emb =incremental_embedding(node,edges_lists[nodes_list.index(node)],H,G,G_model)
		embs[node]=emb
		count+=1
		print(f"{process_name} finished node {node} in {(time.process_time() -start_time):.2f} seconds. {((count/total)*100):.3f}%")
	dfembs=pd.DataFrame.from_dict(embs,orient='index')
	dfembs=dfembs.reset_index(level=0)
	settings.lck.acquire()
	with open(f'{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.INCREMENTAL_MODEL}_{settings.BASE_ALGORITHM}_{settings.YEAR_START+1}.csv', 'a+', newline='') as write_obj:
		dfembs.to_csv(write_obj, index=False, sep=' ', header=False)
	
	settings.lck.release()

	print(f"{process_name} ended")


def incremental_embedding(node: int,edges_list:list,H:Union[nx.DiGraph,nx.Graph],completeGraph:Union[nx.DiGraph,nx.Graph],G_model:KeyedVectors):
	'''It takes a node, a list of edges, a graph, a model and returns the embedding of the node
	
	Parameters
	----------
	node
		the node to be added to the graph
	edges_list
		list of edges of the node to be added
	H
		the graph of the hubs
	completeGraph
		the graph of the year t
	G_model
		the embedding of the graph at time t
	
	Returns
	-------
		The embedding of the node that is being added to the graph.
	
	'''

	filenodetoeliminate= open(f'{settings.DIRECTORY}tmp/nodetoeliminate.csv','a+')
	PATH_LOG=f'{settings.DIRECTORY}logs/log{node}.txt'
	f_log=open(PATH_LOG,'w+')
	isproblematic=False
	try:
		G=nx.from_pandas_edgelist(pd.read_csv(f'{settings.DIRECTORY}edgescumulative/edges{settings.YEAR_START+1}.csv'),source='Source',target='Target')
		tmp = nx.Graph()
		if settings.DIRECTED:
			tmp = nx.DiGraph()
		tmp = read_edges_list_no_file(edges_list,tmp)
		f_log.write('creata lista tmp di archi\n')
		H_plus_node = H.copy()
		H_init_edges_number = len(H_plus_node.edges())
		
		embeddable = False
		#da qui in poi ho fatto molte modifiche
		#controlla se il nodo è collegato direttamente con un hub: caso migliore
		for e in tmp.edges():
			if (e[1]) in H.nodes() or e[1] in H.nodes():
				#if node has a link with someone in Hubs
				H_plus_node.add_edge(e[0],e[1])
				f_log.write(f'aggiunta archi {e[0]} e {e[1]}. {node} è embeddabile\n')
				embeddable = True

		if(H_init_edges_number == len(H_plus_node.edges())):
			#if node has NOT ANY link with someone in Hubs
			f_log.write(f'non è stato trovato un hub connesso con il nodo {node}\n')
			found = False
			it=0
			incident_vertexes=[]
			exist=False

			
			while(not found and it<len(tmp.edges())):
				e = list(tmp.edges())[it]
				for incident_vertex in e:
					f_log.write(f'incident vertex:{incident_vertex}\n')
					#non ha collegamenti con hub, quindi prova a vedere se c'è una path verso un hub attraverso un nodo vicino
					if incident_vertex != node:
						if incident_vertex in G.nodes():
							#vertex linked with node is in G
							f_log.write(f'incident vertex {incident_vertex} è in G\n')
							found = True
							#G.add_edge(e[0],e[1])
							f_log.write(f'Aggiungo arco:{e[0]} {e[1]}\n')

							#scorro la lista degli hub invece di fare una random choice e vedere se ha path
							for hubtmp in H.nodes():
								
								exist = nx.has_path(G, source=node, target=hubtmp)
								f_log.write(f'Esiste path tra nodo e Hub {hubtmp}? {exist}\n')
								#se esiste una path va bene e va direttamente allo step successivo
								if exist:
									break
							#se non esiste rimuovo gli archi che stavo analizzando
							if not exist:
								incident_vertexes.append(incident_vertex)
								found=False
								
							if(exist):
								#niente di nuovo qui. Si fa una shortest path e si aggiunge tutta nel grafo hub
								sh_paths =nx.shortest_path(G, source=node, target=hubtmp, weight=None, method='dijkstra')
								f_log.write(f'Creo shortest path\n')

								#add this walk to H_plus_node
								for i in range(len(sh_paths)):
									if(i+1<len(sh_paths)):
										f_log.write(f'shortest path:{sh_paths[i]} + {sh_paths[i+1]}\n')
										H_plus_node.add_edge(sh_paths[i],sh_paths[i+1])
								embeddable=True
				it+=1
			#caso peggiore: il nodo non ha archi verso hubs, di conseguenza si prova ad aggiungere un arco verso un hub random
			if not exist and len(incident_vertexes)>0:
				filenodetoeliminate.write(f'{node} non ci sono path verso Hubs\n')
				isproblematic=True
				#TODO creo un arco fittizio (sarà giusto?) in modo che ci sia una path tra l'incident vertex e un hub a caso
				hubrandom=random.choice(list(H.nodes()))
				vertextouse=random.choice(incident_vertexes)
				G.add_edge(node,vertextouse)
				print(f'Creazione arco fittizio con {node} + {vertextouse} con {hubrandom}')
				f_log.write(f'Creazione arco fittizio con {node} + {vertextouse} con {hubrandom} in hub_plus_node\n')
				H_plus_node.add_edge(node,vertextouse)
				
				f_log.write(f'aggiungo arco {e[0]} {e[1]} a H_PLUS_NODE\n')
				H_plus_node.add_edge(vertextouse,hubrandom)
				embeddable=True
			#teoricamente caso irraggiungibile col nuovo modo
			elif (not found):
				filenodetoeliminate.write(f'{node} non ci sono archi  con nodi esistenti in G \n')
				isproblematic=True
				hubrandom=random.choice(list(H.nodes()))
				f_log.write(f'Creazione arco fittizio con {node} e hub {hubrandom}\n')
				print(f'Creazione arco fittizio con {node} e hub {hubrandom}')
				
				G.add_edge(node,hubrandom)
				H_plus_node.add_edge(node,hubrandom)
				embeddable=True

		####### AT THIS POINT I'M GOOD WITH H 
		if(embeddable):
			f_log.write(f'{node} è embeddabile\n')
			model_i= None
			
			if settings.BASE_ALGORITHM =="deepwalk":
				f_log.write(f'Deepwalk:\n')
				H_plus_node,model_i=incrementalDeepWalk(H_plus_node, node)
				
			else:
				f_log.write(f'node2vec:\n')
				H_plus_node,model_i=incrementalNode2Vec(H_plus_node, node)
			
			assert model_i
			f_log.write(f'extract embedding con nodo\n')
			model_i_dict = extract_embedding_for_Hub_nodes(H_plus_node,model_i)
			#pre-processing for alignment
			
			e_i_raw = model_i[str(node)]
			
			 
			
			#remove node to be incrematlly added
			f_log.write(f'extract embeddings senza nodo\n')
			H_model=extract_embedding_for_Hub_nodes(H,G_model)
			
			A_embeddings = []
			B_embeddings = []
			neighboors=[]
			
			#takes neighboors of node in Hub graph and add also neighboors of these ones (second order neighboors)
			i_neighboors = list(H_plus_node[node])
			for n in i_neighboors:
				neighboors.append(n)
				second_order = list(H_plus_node[n])
				if(node in second_order):
					second_order.remove(node)
				neighboors= neighboors + second_order
			neighboors = set(neighboors)

			# Creating two lists of embeddings, A_embeddings and B_embeddings.
			#check if neighboors are in dictionary of embeddings of Hubs minus new node or in dictionary of hubs plus node
			#QUESTO È IL VECCHIO CODICE
			if isproblematic:
				for n in neighboors:
					for e in H_model:
						if e == n:
							A_embeddings.append(H_model[e])#e[1:settings.DIMENSION+1])
					for f in model_i_dict:
						if f == n:
							B_embeddings.append(model_i_dict[f])#f[1:settings.DIMENSION+1])
			#QUESTO È UN IPOTETICO FIX
			else:
				for n in list(H.nodes()):
					A_embeddings.append(H_model[n]) #H_model: modello di hubs prima di aggiungere il nuovo nodo (embedding fatto all'anno t)
					B_embeddings.append(model_i_dict[n])#model_i_dict: modello di hubs dopo aver aggiunto il nuovo nodo (embedding fatto all'anno t+1)
				
			#da qui si procede con l'allineamento
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
			
			f_log.close()
			os.remove(PATH_LOG)
			filenodetoeliminate.close()
			return e_i
	except Exception as e:
		f_log.write(f'{str(e)} with node {node}\n')
		f_log.close()
		raise Exception(f'{e} with node {node}')

	
""" def Deepwalk(edges_file,edges_type,embedding_dir,embeddingName,emb_workers,window_size,representation_size,NUM_WALKS,LEN_WALKS, separator=' '):
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
	return model """
def extract_embedding_for_Hub_nodes(H:nx.DiGraph,G_model:Word2Vec)-> dict:
	'''It takes a graph and a word2vec model and returns a dictionary of node embeddings for the nodes in
	the graph
	
	Parameters
	----------
	H : nx.DiGraph
		The graph that we want to embed
	G_model : Word2Vec
		Word2Vec model
	
	Returns
	-------
		A dictionary of the form {node:embedding}
	
	'''
	try:
		H_mod = {}
		for n in H.nodes():
			e_n=G_model[str(n)]
			H_mod[n]=e_n
		return H_mod
	except Exception as e:
		print(str(e))
		raise Exception(f'{e}')
	

def getTorchData(G:nx.DiGraph):
	''' The function takes a networkx graph and returns a PyTorch geometric graph and a dictionary that
	maps the node ids in the PyTorch geometric graph to the node ids in the networkx graph
	
	Parameters
	----------
	G : nx.DiGraph
		nx.DiGraph - the graph we want to convert to a PyTorch geometric graph
	
	Returns
	-------
		A tuple of two elements. The first element is a Data object from the PyTorch Geometric library. The
	second element is a dictionary that maps the node ids to the node names.
	
	'''
		
    # Create a dictionary of the mappings from graph --> node id
	mapping_dict = {x: i for i, x in enumerate(list(G.nodes()))}
	inv_map = {v: k for k, v in mapping_dict.items()}


	# Now create a source, target, and edge list for PyTorch geometric graph
	edge_source_list = []
	edge_target_list = []

	# iterate through all the edges
	for e in G.edges():
	# first element of tuple is appended to source edge list
		edge_source_list.append(mapping_dict[e[0]])

		# last element of tuple is appended to target edge list
		edge_target_list.append(mapping_dict[e[1]]) 


	# now create full edge lists for pytorch geometric - undirected edges need to be defined in both directions

	full_source_list = edge_source_list    # full source list (+edge_target_list)
	full_target_list = edge_target_list     # full target list  (+edge_source_list) 

	# now convert these to torch tensors
	edge_index_tensor = torch.LongTensor( np.concatenate([ [np.array(full_source_list)], [np.array(full_target_list)]] ))
	torchG = Data(edge_index=edge_index_tensor)
	return(torchG, inv_map)

def incrementalDeepWalk(H_plus_node:nx.DiGraph, node:int)->Tuple[nx.DiGraph,KeyedVectors]:
	'''It takes a graph + node and node, and returns the graph and the embedding of the
	graph + node
	
	Parameters
	----------
	H_plus_node : nx.DiGraph
		the graph to be embedded
	node : int
		the node to be added to the graph
	
	Returns
	-------
		the graph and the embedding model.
	
	'''

	intostr={x: str(x) for x in list(H_plus_node.nodes())}
	intostr_inv={str(x): int(x) for x in list(H_plus_node.nodes())}
	H_plus_node = nx.relabel_nodes(H_plus_node, intostr)
	model_i= deepwalk.DeepWalk(H_plus_node,settings.LENGTH_WALKS,workers=1, num_walks=settings.NUM_WALKS)
	model_i.train(embed_size=settings.DIMENSION, window_size=settings.WINDOWS_SIZE, workers=1, node=node)
	model_i=pd.DataFrame.from_dict(model_i.get_embeddings(),orient='index')
	with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START+1}{node}.csv','w+', newline='',encoding='utf-8') as f:
		f.write(f'{model_i.shape[0]} {model_i.shape[1]}\n')
		model_i['id'] = model_i.index
		model_i=model_i.set_index('id')
		model_i.to_csv(f, header=False, index=True,sep=' ')
	model_i = KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START+1}{node}.csv')
	os.remove(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START+1}{node}.csv')
	H_plus_node = nx.relabel_nodes(H_plus_node, intostr_inv)
	return H_plus_node,model_i

def incrementalNode2Vec(H_plus_node:nx.DiGraph, node:int)->Tuple[nx.DiGraph,KeyedVectors]:
	'''It takes a graph and a node, and returns the graph with the node added and the embedding of the node
	
	Parameters
	----------
	H_plus_node : nx.DiGraph
		the graph to be embedded
	node : int
		the node to be added to the graph
	
	Returns
	-------
		The embedding of the graph and the graph with the new node
	
	'''
	torchH,inv_map=getTorchData(G=H_plus_node)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = nn.Node2Vec(torchH.edge_index, embedding_dim=settings.DIMENSION, walk_length=settings.LENGTH_WALKS,
		context_size=settings.WINDOWS_SIZE, walks_per_node=settings.NUM_WALKS,
		num_negative_samples=1, p=1, q=1, sparse=True).to(device)
	print(f'train: {node}')
	loader = model.loader(batch_size=128, shuffle=True, num_workers=1)
	optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
	def train():
		model.train()
		total_loss = 0
		for pos_rw, neg_rw in loader:
			optimizer.zero_grad()
			loss = model.loss(pos_rw.to(device), neg_rw.to(device))
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		return total_loss / len(loader)
	for epoch in range(1, 30):
		loss = train()
		#acc = test()
	embH=pd.DataFrame(model.forward().tolist())
	with open(f"{settings.DIRECTORY}{settings.EMBEDDING_DIR}bin/{settings.NAME_DATA}{settings.YEAR_START+1}TORCH{node}.csv",'w+', newline='',encoding='utf-8') as f:
		f.write(f'{embH.shape[0]} {embH.shape[1]}\n')
		embH['id'] = embH.index
		embH=embH.replace({"id": inv_map})
		embH=embH.set_index('id')
		embH.to_csv(f, header=False,index=True,sep=' ')
	model_i = KeyedVectors.load_word2vec_format(f"{settings.DIRECTORY}{settings.EMBEDDING_DIR}bin/{settings.NAME_DATA}{settings.YEAR_START+1}TORCH{node}.csv")
	os.remove(f"{settings.DIRECTORY}{settings.EMBEDDING_DIR}bin/{settings.NAME_DATA}{settings.YEAR_START+1}TORCH{node}.csv")
	torch.cuda.empty_cache()
	return H_plus_node,model_i