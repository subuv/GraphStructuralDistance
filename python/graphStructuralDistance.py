from __future__ import division
import logging, sys
from igraph import *
import numpy as np
import math
from collections import defaultdict
import itertools
import datetime
from random import randint
from simanneal import Annealer

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def gdist(ori_arr, anon_arr):
		gdist = 0

		#logging.info("Computing graph distance...")
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

class StructDist(Annealer):
	def __init__(self, state, A1, A2, d0):
		self.A1 = A1
		self.A2 = A2
		self.A3 = A1.copy()
		self.A4 = A2.copy()
		self.n_ori, self.m_ori = A1.shape
		self.n_anon, self.m_anon = A2.shape
		self.r1 = 0
		self.r2 = 0
		self.d0 = d0
		self.seen = set()
		super(StructDist, self).__init__(state)  # important! 

	def move(self):
		"""Swaps two cities in the route."""
		temp_r1 = randint(0, self.n_ori-1)
		temp_r2 = randint(0, self.n_anon-1)
		count = 0
		while (temp_r1, temp_r2) in self.seen:
			count = count + 1
			temp_r1 = randint(0, self.n_ori-1)
			temp_r2 = randint(0, self.n_anon-1)
			if count > self.n_ori:
				break
		
		self.r1 = temp_r1
		self.r2 = temp_r2

		self.seen.add((temp_r1, temp_r2))

		temp_A3 = self.A1.copy()
		temp_A4 = self.A2.copy()

		temp_A4[:,[temp_r1, temp_r2]] = temp_A4[:,[temp_r2, temp_r1]]
		self.A3, self.A4 = temp_A3, temp_A4

	def energy(self):
		"""Calculates the length of the route."""
		#logging.info("d0 is %s" % self.d0)
		d1 = self.d0 + gddelta(self.A3, self.A4, self.r1, self.r2)
		#logging.info("d1 is %s" % d1)
		
		if d1 < self.d0:
			db = d1
		else:
			db = self.d0

		#nc = sc.special.binom(list(range(1, 64)), 2)
		#nc = map(int,nc)

		#logging.info("db is %s" % db)
		
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

	init_state = A2.copy()

	tsp = StructDist(init_state, A1, A2, gdist(A1, A2))
	#auto_schedule = tsp.auto(minutes=0.1)
	# since our state is just a list, slice is the fastest way to copy
	tsp.copy_strategy = "slice"
	#tsp.set_schedule(auto_schedule)
	tsp.Tmax = 98.6  # Max (starting) temperature
	tsp.Tmin = 0.1      # Min (ending) temperature
	tsp.steps = 1000000   # Number of iterations
	tsp.updates = 10   # Number of updates (by default an update prints to stdout)
	state, e = tsp.anneal()

	print e
main()
