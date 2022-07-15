#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
import pandas as pd

logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]
      #print(path)
	  
    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]

# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()
  
  total = 0 
  with open(file_) as f:
    for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
      adjlist.extend(adj_chunk)
      total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=False,separator=' '):
  G = Graph()
  if separator==' ':
    edges= pd.read_csv(file_,sep=separator, header=None)
    for row in edges.itertuples():
      G[row[1]].append(row[2])
  else:
    edges= pd.read_csv(file_,sep=separator)
    edges=edges[['Source','Target']]
    for row in edges.itertuples():
      G[row.Source].append(row.Target)
  """ with open(file_) as f:
    f.readline()
    for l in f:
      x = l.split(separator)[0]
      y = l.split(separator)[1].replace("\n","")
      #x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)
      G[x].append(y)
      if undirected:
        G[y].append(x) """
  
  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=False):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G


######################## WALKS .PY 

import logging
from io import open
from os import path
from time import time
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from six.moves import zip


logger = logging.getLogger("deepwalk")

__current_graph = None

# speed up the string encoding
__vertex2str = None

def count_words(file):
  """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  c = Counter()
  with open(file, 'r') as f:
    for l in f:
      words = l.strip().split()
      c.update(words)
  return c


def count_textfiles(files, workers=1):
  c = Counter()
  with ProcessPoolExecutor(max_workers=workers) as executor:
    for c_ in executor.map(count_words, files):
      c.update(c_)
  return c


def count_lines(f):
  if path.isfile(f):
    num_lines = sum(1 for line in open(f))
    return num_lines
  else:
    return 0

def _write_walks_to_disk(args,):
  num_paths, path_length, alpha, rand, f = args
  global __current_graph
  G =__current_graph 
  
  
  t_0 = time()
  with open(f, 'w') as fout:
    
    for walk in build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                 alpha=alpha, rand=rand):
      fout.write(u"{}\n".format(u" ".join(v for v in walk)))
  logger.debug("Generated new file {}, it took {} seconds".format(f, time() - t_0))
  return f

def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=1,
                        always_rebuild=True):
  global __current_graph
  __current_graph = G
  
  files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_paths))]
  expected_size = len(G)
  args_list = []
  files = []

  if num_paths <= num_workers:
    paths_per_worker = [1 for x in range(num_paths)]
  else:
    paths_per_worker = [len(list(filter(lambda z: z!= None, [y for y in x])))
                        for x in grouper(int(num_paths / num_workers)+1, range(1, num_paths+1))]

  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
      if always_rebuild or size != (ppw*expected_size):
        args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2**31)), file_))
      else:
        files.append(file_)
		
  for file_ in map(_write_walks_to_disk, args_list):
    files.append(file_)
	  
  #with ProcessPoolExecutor(max_workers=num_workers) as executor:
    
    
    
    

  return files

class WalksCorpus(object):
  def __init__(self, file_list):
    self.file_list = file_list
  def __iter__(self):
    for file in self.file_list:
      with open(file, 'r') as f:
        for line in f:
          yield line.split()

def combine_files_iter(file_list):
  for file in file_list:
    with open(file, 'r') as f:
      for line in f:
        yield line.split()
