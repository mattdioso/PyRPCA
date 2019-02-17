import numpy as np 
import scipy as spy 
import math
import os

def bdsqr(alpha, beta):
# BDSQR: Compute the singular values and bottom element of
#        the left singular vectors of a (k+1) x k lower bidiagonal 
#        matrix with diagonal alpha(1:k) and lower bidiagonal beta(1:k),
#        where length(alpha) = length(beta) = k.
#
# [sigma,bnd] = bdsqr(alpha,beta)
#
# Input parameters:
#   alpha(1:k)   : Diagonal elements.
#   beta(1:k)    : Sub-diagonal elements.
# Output parameters:
#   sigma(1:k)  : Computed eigenvalues.
#   bnd(1:k)    : Bottom elements in left singular vectors.

# Below is a very slow replacement for the BDSQR MEX-file.

# warning('PROPACK:NotUsingMex','Using slow matlab code for bdsqr.')

	k = len(alpha)
	if math.min(np.shape(alpha).conj().transpose()) != 1 or math.min(np.shape(beta).conj().transpose()) != 1:
		print("[bdsqr] alpha and beta must be vectors")
		return
	elif len(beta) != k:
		print("[bdsqr] alpha and beta must have the same length")
		return

	B = np.spdiags([alpha[:], beta[:]], [0, -1], k+1, k)
	U, S, V = np.linalg.svd(full(B), 0)
	sigma = np.diag(S)
	bnd = U[end, 1:k].conj().transpose()
	return sigma, bnd