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
	global DIRECTORY
	global YEAR_CURRENT
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
	NAME_DATA = DATA.CORA.value
	YEAR_START=1985 if NAME_DATA=='CORA' else 2009
	YEAR_CURRENT= 1
	DIRECTED = False
	YEAR_MAX= 13 if NAME_DATA=='CORA' else 3
	CENTRALITY= Centralities.BETWEENNESS.value
	DATA = f"edges/{NAME_DATA}{YEAR_START}.csv"
	INCREMENTAL_DIR=f"{NAME_DATA}_incremental/"
	
	INCREMENTAL_MODEL = f'{NAME_DATA}_incremental'
	EMBEDDING_DIR = "embeddings/"
	CUT_THRESHOLD=30
	WINDOWS_SIZE=10
	DIMENSION=128
	NUM_WALKS=80
	LENGTH_WALKS=10
	BASE_ALGORITHM=ALGORITHM.DEEPWALK.value
	STATIC_ALGORITHM= ALGORITHM.CTDNE.value
	TMP ="tmp/"
	DIRECTORY=f'cora/{CENTRALITY}/' if NAME_DATA == 'CORA' else f"arxiv/{CENTRALITY}/"
