import os
import numpy as np 
import math
from globalVariables import *

def solver_RPCA_SPGL1(AY, lambda_S, epsilon, A_cell, opts):
# [L,S,errHist,tau] = solver_RPCA_SPGL1(Y,lambda_S, epsilon, A_cell, opts)
# Solves the problem
#   minimize_{L,S} f(L,S)
#   subject to 
#       || A( L+S ) - Y ||_F <= epsilon
#   if opts.sum = true
#       (1) f(L,S) = ||L||_* + lambda_S ||S||_1 
#   if opts.max = true
#       (2) f(L,S) = max( ||L||_* , lambda_S ||S||_1 ) 
#
#   By default, A is the identity, 
#   or if A_cell is provided, where A_cell = {A, At}
#   (A is a function handle, At is a function handle to the transpose of A)
#
#   This uses the trick from SPGL1 to call several subproblems
#
#   errHist(1,:) is a record of the residual
#   errHist(2,:) is a record of the full objective (that is, .5*resid^2 )
#   errHist(3,:) is the output of opts.errFcn if provided
#
# opts is a structure with options:
#   opts.tau0       starting point for SPGL1 search (default: 10)
#   opts.SPGL1_tol  tolerance for 1D Newton search (default: 1e-2)
#   opts.SPGL1_maxIts   maximum number of iterations for 1D search (default: 10)
#
# And these options are the same as in solver_RPCA_constrained.m
#   opts.sum, opts.max  (as described above)
#   opts.L0         initial guess for L (default is 0)
#   opts.S0         initial guess for S (default is 0)
#   opts.tol        sets stopping tolerance (default is 1e-6)
#   opts.maxIts     sets maximum number of iterations
#   opts.printEvery will print information this many iterations
#   opts.displayTime will print out timing information (default is true for large problems)
#   opts.errFcn     a function of (L,S) that records information
#   opts.trueObj    if provided, this will be subtracted from errHist(2,:)
#   opts.Lip        Lipschitz constant, i.e., 2*spectralNorm(A)^2
#                       by default, assume 2 (e.g., good if A = P_Omega)
#   opts.FISTA      whether to use FISTA or not. By default, true
#     opts.restart  how often to restart FISTA; set to -Inf to make it automatic
#   opts.BB         whether to use the Barzilai-Borwein spectral steplength
#     opts.BB_type  which BB stepsize to take. Default is 1, the larger step
#     opts.BB_split whether to calculate stepslengths for S and L independently.
#       Default is false, which is recommended.
#   opts.quasiNewton  uses quasi-Newton-like Gauss-Seidel scheme.
#                     Only available in "max" mode
#     opts.quasiNewton_stepsize     stepsize length. Default is .8*(2/Lip)
#     opts.quasinewton_SLS          whether to take S-L-S sequence (default is true)
#                                   otherwise, takes a L-S Gauss-Seidel sequence
#   opts.SVDstyle   controls what type of SVD is performed.
#       1 = full svd using matlab's "svd". Best for small problems
#       2 = partial svd using matlab's "svds". Not recommended.
#       3 = partial svd using PROPACK, if installed. Better than option 2, worse than 4
#       4 = partial svd using randomized linear algebra, following
#           the Halko/Tropp/Martinnson "Structure in Randomness" paper
#       in option 4, there are additional options:
#       opts.SVDwarmstart   whether to "warm-start" the algorithm
#       opts.SVDnPower  number of power iterations (default is 2 unless warm start)
#       opts.SVDoffset  oversampling, e.g., "rho" in Tropp's paper. Default is 5
#
#   opts.L1L2      instead of using l1 penalty, e.g., norm(S(:),1), we can
#       also use block norm penalties, such as (if opts.L1L2 = 'rows')
#       the sum of the l2-norm of rows (i.e., l1-norm of rows),
#       or if opts.L1L2='cols', the sum of the l2-norms of colimns.
#       By default, or if opts.L1L2 = [] or false, then uses usual l1 norm.
#       [Feature added April 17 2015]
#
	if len(locals()) < 5:
		opts =[]

	def setOpts(field, default):
		if not field in opts:
			opts[field] = default
		out = opts[field]
		opts.pop(field, None)
		return out

	if len(locals()) < 4 or A_cell.isEmpty:
		A = lambda X: X(:)
		n1, n2 = AY.shape
		At = lambda x : np.reshape(x, (n1, n2))
	else:
		A = A_cell[0]
		At = A_cell[1]
		dim = AY.shape
		if dim[1] > 1:
			AY = A(AY)

	tauInitial = setOpts('tau0', math.exp(1))

	LIL2 = setOpts('LIL2', 0)

	if LIL2.isEmpty:
		LIL2=0

	if LIL2:
		if LIL2.lower().find('row'):
			LIL2 = 'rows'
		else if LIL2.lower().find('col'):
			LIL2 = 'cols'
		else:
			print('unrecognized option for LIL2: should be row or column or 0')
			return

	opts[LIL2] = LIL2

	finalTol = setOpts('tol', math.exp(-6))
	sumProject = setOpts('sum', false)
	maxProject = setOpts('max', false)
	if sumProject && maxProject || not sumProject && not maxProject:
		print("Must choose either sum or max type projection")
		return
	opts[max] = maxProject
	opts[sum] = sumProject
	rNormOld = Inf
	tau = tauInitial
	SPGL1_tol = setOpts('SPGL1_tol', math.exp(-2))
	SPGL1_maxIts = setOpts('SPGL1_maxIts', 10)
	errHist = []

	for nSteps in range(0, SPGL1_maxIts):
		opts[tol] = math.max(finalTol, finalTol*10^((4/nSteps)-1))
		print("\n==Running SPGL1 Newton iteration with tau=%.2f\n\n", tau)
		L, S, errHistTemp = solver_RPCA_constrained(AY, lambda_S, tau, A_cell, opts)

		errHist = errHist + errHistTemp

		rNorm = errHist(end, 1)

		if math.abs(epsilon -rNorm) < SPGL1_tol*epsilon:
			print("reached end of SPGL1 iterations: converged to right residual")
			break

		if math.abs(rNormOld - rNorm) < .1 * SPGL1_tol:
			print("reached end of SPGL1 iterations: converged")
			break

		G = At(A(L+S-AY))

		if sumProject:
			normG = math.max(np.linalg.norm(G), (1/lambda_S)*np.linalg.norm(G(:), inf))
		else if maxProject:
			if not any(LIL2):
				normG = np.linalg.norm(G) + (1/lambda_S)*np.linalg.norm(G(:), inf)
			else if LIL2 == 'rows':
				normG = np.linalg.norm(G) + (1/lambda_S)*np.linalg.norm(math.sqrt(np.sum(np.power(G, 2), axis=1), inf))
			else if LIL2 == 'cols':
				normG = np.linalg.norm(G) + (1/lambda_S)*np.linalg.norm(math.sqrt(np.sum(np.power(G, 2), axis=0), inf))

		phiPrime = -normG/rNorm
		ALPHA = .99
		tauOld = tau
		tau = tau + ALPHA*(epsilon - rNorm)/phiPrime
		tau = math.min(math.max(tau, tauOld/10), math.exp(10))

		opts[S0] = S
		opts[L0] = L

	if finalTol < math.exp(-6):
		opts[tol] = finalTol/10
		opts[S0] = S 
		opts[L0] = L
		opts[SVDnPower] = 3
		L, S, errHistTemp = solver_RPCA_constrained(AY, lambda_S, tau, A_cell, opts)
		errHist = errHist + errHistTemp

	return L, S, errHist, tau
