from enum import Enum


def init():
	global DIRECTED
	global DATA
	global INCREMENTAL_DIR
	global NAME_DATA
	global EMBEDDING_DIR
	global CUT_THRESHOLD
	global WINDOWS_SIZE
	global DIMENSION
	global NUM_WALKS
	global LENGTH_WALKS
	global BASE_ALGORITHM
	global TMP
	global YEAR_START
	global STATIC_ALGORITHM
	global INCREMENTAL_MODEL
	global lck
	import threading
	global STATIC_REMBEDDING
	global DIRECTORY
	global YEAR_CURRENT
	global FOLDER
	global HALF_YEAR
	global COUNT
	global SPLIT_NODES
	global YEAR_MAX
	global CENTRALITY
	lck = threading.Lock()
	class ALGORITHM(Enum):
		NODE2VEC='node2vec'
		DEEPWALK='deepwalk'
		TNODEEMBEDDING='tnodeembedding'
		CTDNE='ctdne'
	class Centralities(Enum):
		BETWEENNESS='betweenness'
		DEGREE='degree'
		EIGENVECTOR='eigenvector'
		PAGERANK='pagerank'
	
	class DATA(Enum):
		CORA='CORA'
		ARXIV='ARXIV'
	NAME_DATA = DATA.ARXIV.value
	#1985
	YEAR_START=1985 if NAME_DATA=='CORA' else 2009
	
	DIRECTED = True
	STATIC_REMBEDDING=False
	SPLIT_NODES=False
	#13
	YEAR_MAX= 13 if NAME_DATA=='CORA' else 3 #4 solo per static rembedding
	#betweenness!->degree->eigenvector->pagerank
	CENTRALITY= Centralities.DEGREE.value
	DATA = f"edges/{NAME_DATA}{YEAR_START}.csv"
	INCREMENTAL_DIR=f"{NAME_DATA}_incremental/"
	FOLDER=f'cora/' if NAME_DATA=='CORA' else f'arxiv/'
	INCREMENTAL_MODEL = f'{NAME_DATA}_incremental'
	EMBEDDING_DIR = "embeddings/"
	CUT_THRESHOLD=30
	WINDOWS_SIZE=10
	DIMENSION=128
	NUM_WALKS=80
	LENGTH_WALKS=10
	COUNT=1 #21
	if SPLIT_NODES:
		if COUNT%2==0:
			YEAR_CURRENT, HALF_YEAR=COUNT//2, 1
		else: YEAR_CURRENT, HALF_YEAR=(COUNT//2)+1, 0
	else: YEAR_CURRENT,HALF_YEAR=COUNT,-1
	BASE_ALGORITHM=ALGORITHM.DEEPWALK.value
	STATIC_ALGORITHM= ALGORITHM.CTDNE.value
	TMP ="tmp/"
	if NAME_DATA=='CORA':
		if SPLIT_NODES:
			DIRECTORY=f'cora/{CENTRALITY}/split/'
		else:
			DIRECTORY=f'cora/{CENTRALITY}/nosplit/'
	else:
		if SPLIT_NODES:
			DIRECTORY=f'arxiv/{CENTRALITY}/split/'
		else:
			DIRECTORY=f'arxiv/{CENTRALITY}/nosplit/'
	
