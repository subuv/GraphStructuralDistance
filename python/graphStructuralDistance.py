from __future__ import division
import pandas as pd
import logging, sys
from igraph import *
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
from scipy.stats.stats import pearsonr
from joblib import Parallel, delayed
from scipy.optimize import basinhopping
from collections import defaultdict
import itertools
import datetime
from random import randint

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def gdist(ori_arr, anon_arr):
	gdist = 0

	logging.info("Computing graph distance...")
	#sum(sum(map(lambda i: map(lambda x, y: np.absolute(x - y), ori_arr[i], anon_arr[i]), xrange(len(ori_arr))),[]))
	gdist = sum(sum(np.absolute(ori_arr - anon_arr)))

	return gdist

def gddelta(A1, A2, a, b):
	gddelta = 0

	#logging.info("Computing graph delta distance...")

	[n_ori, m_ori] = A1.shape

	for i in range(0, n_ori):
		if i not in (a, b):
			gddelta = gddelta + (np.absolute(A1[a, i] - A2[b, i]) - np.absolute(A1[a, i] - A2[a, i])) + (np.absolute(A1[b, i] - A2[a, i]) - np.absolute(A1[b, i] - A2[b, i])) + (np.absolute(A1[i, a] - A2[i, b]) - np.absolute(A1[i, a] - A2[i, a])) + (np.absolute(A1[i, b] - A2[i, a]) - np.absolute(A1[i, b] - A2[i, b]))

	gddelta = gddelta + (np.absolute(A1[a, a] - A2[b, b]) - np.absolute(A1[a, a] - A2[a, a])) + (np.absolute(A1[b, b] - A2[a, a]) - np.absolute(A1[b, b] - A2[b, b])) + (np.absolute(A1[a, b] - A2[b, a]) - np.absolute(A1[a, b] - A2[a, b])) + (np.absolute(A1[b, a] - A2[a, b]) - np.absolute(A1[b, a] - A2[b, a]))
	
	return gddelta

seen = set()

def sdist(db, A1, A2, d0):
	[n_ori, m_ori] = A1.shape
	[n_anon, m_anon] = A2.shape
	
	r1 = randint(0, n_ori-1)
	r2 = randint(0, n_anon-1)
	count = 0
	while (r1, r2) in seen:
		count = count + 1
		r1 = randint(0, n_ori-1)
		r2 = randint(0, n_anon-1)
		if count > n_ori:
			return d0

	seen.add((r1, r2))

	#ddelta = gddelta(A1, A2, r1, r2)
	A3 = A1.copy()
	A4 = A2.copy()

	A4[:,[r1, r2]] = A4[:,[r2, r1]]

	logging.info("d0 is %s" % d0)
	d1 = gdist(A3, A4)
	logging.info("d1 is %s" % d1)
	
	if d1 < d0:
		db = d1
	else:
		db = d0

	#nc = sc.special.binom(list(range(1, 64)), 2)
	#nc = map(int,nc)

	logging.info("db is %s" % db)
	return db

def main():
	print "SDIST: Let us calculate the GDIST of the graph"
	ori_path = raw_input("Path of the graph file 1?\n")
	anon_path = raw_input("Path of the graph file 2?\n")

	ori_igraph = Graph.Read_Ncol(ori_path, directed=True)
	logging.info('Parsing the original graph to get the number of nodes')
	
	n_ori = ori_igraph.vcount()
	logging.info(n_ori)
	e_ori = ori_igraph.ecount()
	logging.info(e_ori)
	
	anon_igraph = Graph.Read_Ncol(anon_path, directed=True)
	logging.info('Parsing the anonymous graph to get the number of nodes')
	
	n_anon = anon_igraph.vcount()
	logging.info(n_anon)
	e_anon = anon_igraph.ecount()
	logging.info(e_anon)
	
	x = randint(0, n_ori)
	y = randint(0, n_anon)

	logging.info("Random label x is %s" % x)
	logging.info("Random label y is %s" % y)

	ori_adj = ori_igraph.get_adjacency(eids=False)
	anon_adj = anon_igraph.get_adjacency(eids=False)

	A1 = np.array(ori_adj.data)
	anon_arr = np.array(anon_adj.data)
	logging.info("Converted adjacency matrix to array")
	
	npad = ((0, np.absolute(n_ori - n_anon)), (0, np.absolute(n_ori - n_anon)))
	A2 = np.pad(anon_arr, pad_width=npad, mode='constant', constant_values=0)
	logging.info("Resized both matrices to same shape")

	x0 = list()
	h_distance = gdist(A1, A2)
	x0.append(h_distance*2)
	minimizer_kwargs = {"method": "COBYLA", "args": (A1, A2, h_distance)}
	ret = basinhopping(sdist, x0, minimizer_kwargs=minimizer_kwargs, niter=500, niter_success=n_ori**2)
	
	print ret

main()
