from lib2to3.pgen2.token import NAME


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
	lck = threading.Lock()
	
	NAME_DATA = "edges"
	YEAR_START=1996
	DIRECTED = True
	DATA = f"edges/{NAME_DATA}{YEAR_START}.csv"
	INCREMENTAL_DIR=f"{NAME_DATA}_incremental/"
	
	INCREMENTAL_MODEL = NAME_DATA+"_incremental.csv"
	EMBEDDING_DIR = "embeddings/"
	CUT_THRESHOLD=19
	WINDOWS_SIZE=10
	DIMENSION=128
	NUM_WALKS=80
	LENGTH_WALKS=10
	BASE_ALGORITHM="deepwalk"
	TMP ="tmp/"