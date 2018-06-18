import numpy as np 
import math
import os
from globalVariables import *
from solver_RPCA_SPGL1 import solver_RPCA_SPGL1

def run_RPCA(X, r, c, params = RPCA_parameters) :
	X = double(X)
	nFrames = X.shape[1]
	L0 = np.kron(ones((1, nFrames)), np.median(X, axis = 1))
	S0 = X - L0

	if len(locals()) < 4:
		lamda = 2 * math.exp(1) -2
		epsilon = 5 * math.exp(1) -3 * np.linalg.norm(X, 'fro')
		opts = {'sum' : false, 'L0' : L0, 'S0' : S0, 'max' : true, 'tau0' : 3 * math.exp(5), 'SPGL1_tol' : math.exp(1), 'tol' : math.exp(1) - 3}
	else:
		lamda = params.lamda
		epsilon = params.epsilon
		opts = {'sum' : params.sum, 'L0' : L0, 'S0' : S0, 'max', params.max, 'tau0' : params.tau0, 'SPGL1_tol' : params.SPGL1_tol, 'tol' : params.tol}

	L, S = solver_RPCA_SPGL1(X, lamda, epsilon, [], opts)

	X = X.asType(np.int8)
	L = L.asType(np.float32)
	S = S.asType(np.float32)

	results = {'X' : X, 'L' : L, 'S' : S, 'dimenstions': [r,c], 'setSize', nFrames}
	return results