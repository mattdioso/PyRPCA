import os
import numpy as np 
import scipy
import math

def randomizedSVD(X, r, rEst, nPower = 1, seed, opts=[]):
# [U,S,V] = randomizedSVD( X, r, rEst, nPower, seed, opts )
#   returns V, S such that X ~ U*S*V' ( a m x n matrix)
#   where S is a r x r matrix
#   rEst >= r is the size of the random multiplies (default: ceil(r+log2(r)) )
#   nPower is number of iterations to do the power method
#    (should be at least 1, which is the default)
#   seed can be the empty matrix; otherwise, it will be used to seed
#       the random number generator (useful if you want reproducible results)
#   opts is a structure containing further options, including:
#       opts.warmStart  Set this to a matrix if you have a good estimate
#           of the row-space of the matrix already. By default,
#           a random matrix is used.
#
#   X can either be a m x n matrix, or it can be a cell array
#       of the form {@(y)X*y, @(y)X'*y, n }
#
# Follows the algorithm from [1]
#
# [1] "Finding Structure with Randomness: Probabilistic Algorithms 
# for Constructing Approximate Matrix Decompositions"
# by N. Halko, P. G. Martinsson, and J. A. Tropp. SIAM Review vol 53 2011.
# http://epubs.siam.org/doi/abs/10.1137/090771806
#

# added to TFOCS in October 2014

	if X.isnumeric():
		X_forward = lambda y : X * y
		X_transpose = lambda y : np.transpose(X)*y
		n = X.shape(1)

	def setOpts(field, default):
		if field not in opts:
			out = default
		else:
			out = opts[field]

		return out

	if rEst.isEmpty():
		rEst = math.ceil(r+np.log2(r))
	rEst = math.min(rEst, n)

	if r>n:
		print("[randomizedSVD] Warning: r > # rows, so truncating it")
		r = n

	if X.isnumeric():
		m = X.shape(0)
		rEst = math.min(rEst, m)
		r = math.min(r, m)

		if r == math.min(m, n):
			U, S, V = np.linalg.svd(X)
			return

	if nPower < 1:
		print("[randomizedSVD] nPower must be >= 1")
		return

	warmStart = setOpts('warmStart', [])
	if not warmStart:
		Q = np.random.randn(n, rEst)
	else:
		Q = warmStart
		if Q.shape(0) != n:
			print("bad height dimension for warmStart")
			return
		if Q.shape(1) > rEst:
			print("[randomizedSVD:warmStartLarge] Warning: warmstart has more columns than rEst")
		else:
			Q = [Q, np.random.randn(n, rEst - Q.shape(1))]

	Q = X_forward(Q)

	for j in range(0, nPower-1):
		Q, R = np.linalg.qr(Q, 0)
		Q = X_transpose(Q)
		Q, R = np.linalg.qr(Q, 0)
		Q = X_forward(Q, 0)

	Q, R = np.linalg.qr(Q, 0)

	V = X_transpose(Q)

	V, R = np.linalg.qr(V, 0)
	U, S, VV = np.linalg.svd(np.transpose(R))
	U = Q*U
	V=V*VV

	U = U[:, 0:r]
	V = V[:, 0:r]
	S = S[0:r, 0:r]

	return U, S, V