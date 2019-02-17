import numpy as np 
import os

def threshold_RPCA(results, C=1):
	results[T] = np.zeros((results[0].shape * results[1].shape, results[setSize]))

	sparse_sigma = lambda x : numpy.std(results[S][:,x])

	for k in range(0, results[setSize]):
		results[T][:,k] = (np.abs(results[S][:,k]) > C*sparse_sigma(k)) * 255

	for i in range(0, results[0].shape * results[1].shape):
		for j in range(0, results[setSize]):
			if results[T][i][j] > 0:
				results[T][i][j] == 1
			else:
				results[T][i][j] == 0

	return results
