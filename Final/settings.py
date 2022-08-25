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
	global INCREMENTAL_MODEL
	global lck
	import threading
	global DIRECTORY
	lck = threading.Lock()
	class ALGORITHM(Enum):
		NODE2VEC='node2vec'
		DEEPWALK='deepwalk'
	
	class DATA(Enum):
		CORA='CORA'
		ARXIV='ARXIV'
	NAME_DATA = DATA.CORA.value
	YEAR_START=1996 if NAME_DATA=='CORA' else 2010
	DIRECTED = True
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
	TMP ="tmp/"
	DIRECTORY='cora/' if NAME_DATA == 'CORA' else "arxiv/"