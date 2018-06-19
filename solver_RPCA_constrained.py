import os
import numpy as np 
import math
from globalVariables import *
import time
import scipy

def solver_RPCA_constrained(AY, lambda_S, tau, A_cell, opts=[]):
# [L,S,errHist] = solver_RPCA_constrained(Y,lambda_S, tau, A_cell, opts)
# Solves the problem
#   minimize_{L,S} .5|| L + S - Y ||_F^2 
#   subject to 
#   if opts.sum = true
#       (1)  ||L||_* + lambda_S ||S||_1 <= tau
#   if opts.max = true
#       (2) max(  ||L||_* , lambda_S ||S||_1 ) <= tau
#
#   if opts.max and opts.sum are false and tau is a negative number, then
#   we solve the problem:
#       minimize_{L,S} .5|| L + S - Y ||_F^2  + abs(tau)*( ||L||_* + lambda_S ||S||_1 )
#   (but see solver_RPCA_Lagrangian.m for a simpler interface)
#
#   or if A_cell is provided, where A_cell = {A, At}
#   (A is a function handle, At is a function handle to the transpose of A)
#   then
#
#   minimize_{L,S} .5|| A(L + S) - Y ||_F^2 
#       subject to ...
#   (here, Y usually represents A(Y); if Y is not the same size
#    as A(L), then we will automatically set Y <-- A(Y) )
#
#   errHist(:,1) is a record of the residual
#   errHist(:,2) is a record of the full objective (that is, .5*resid^2 )
#   errHist(:,3) is the output of opts.errFcn if provided
#
# opts is a structure with options:
#   opts.sum, opts.max  (as described above)
#   opts.L0         initial guess for L (default is 0)
#   opts.S0         initial guess for S (default is 0)
#   opts.size       [n1,n2] where L and S are n1 x n2 matrices. The size is automatically
#       determined in most cases, but when providing a linear operator
#       it may be necessary to provide an explicit size.
#   opts.tol        sets stopping tolerance
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
#  Features that may be added later: [email developers if these are
#    important to you]
#       - Allow Huber loss function.

	# if len(locals()) < 5:
	# 	opts = []

	def setOpts(field, default):
		if not field in opts:
			opts[field] = default
		out = opts[field]
		opts.pop(field, None)
		return out

	if len(locals()) < 4 or A_cell.isEmpty:
		A = lambda X : X[:]
		n1, n2 = AY.shape
		At = lambda x: np.reshape(x, (n1, n2))
	else:
		A = A_cell[0]
		At = A_cell[1]
		dim = AY.shape
		if dim[1] > 1:
			AY = A(AY)
			n1, n2 = AY.shape
		else:
			sz = setOpts('size', [])
			if sz.isEmpty:
				print("cannot determine the size of the variables. please specific opts.size=[n1, n2]")
				return
			n1 = sz[0]
			n2 = sz[1]

	normAY = np.linalg.norm(AY[:])

	SMALL = (n1*n2 <= 50**2)
	MEDIUM = (n1*n2 <= 200**2) and not SMALL 
	LARGE = (n1*n2 <= 1000**2) and not SMALL and not MEDIUM
	HUGE = (n1*n2 > 1000**2)

	tol = setOpts('tol', math.exp(-6)*(SMALL or MEDIUM) + math.exp(-4)*LARGE + math.exp(-3)*HUGE)
	maxIts = setOpts('maxIts', math.exp(3)*(SMALL or MEDIUM) +  400*LARGE + 200*HUGE)
	printEvery = setOpts('printEvery', 100*SMALL + 50*MEDIUM + 5*LARGE + 1*HUGE)
	errFcn = setOpts('errFcn', [])
	Lip = setOpts('Lip', 2)
	restart = setOpts('restart', -math.inf)
	trueObj = setOpts('trueObj', 0)
	sumProject = setOpts('sum', false)
	maxProject = setOpts('max', false)

	if tau < 0:
		Lagranian = true
		if sumProject or maxProject:
			print("in Lagranian mode (when tau<0 signifies lambda=|tau|), turn off sum/maxProject")
			return
		lamda = math.abs(tau)
		tau=[]
	else:
		Lagranian = false
		if (sumProject and maxProject) or (not sumProject and not maxProject):
			print("must choose either 'sum' or 'max' type projection")
			return

	QUASINEWTON = setOpts('quasiNewton', maxProject or Lagranian)
	FISTA = setOpts('FISTA', not QUASINEWTON)
	BB = setOpts('BB', not QUASINEWTON)
	if BB and FISTA:
		print("solver_RPCA: convergence, Convergence not guaranteed with FISTA if opts[BB]=true")

	BB_split = setOpts('BB_split', false)
	BB_type = setOpts('BB_type', 1)
	stepsizeQN = setOpts('quasiNewton_stepsize', .8*2/Lip)
	S_L_S = setOpts('quasinewton_SLS', true)
	displayTime = setOpts('displayTime', LARGE or HUGE)
	SVDstyle = setOpts('SVDstyle', 1*SMALL + 4*(not SMALL))

	SVDwarmstart = setOpts('SVDwarmstart', true)
	SVDnPower = setOpts('SVDnPower', 1 + (not SVDwarmstart))
	SVDoffset = setOpts('SVDoffset', 5)
	SVDopts = {'SVDstyle' : SVDstyle, 'warmstart' : SVDwarmstart, 'nPower' : SVDnPower, 'offset' : SVDoffset}

	if QUASINEWTON:
		if sumProject:
			print("cannot run quasi-newton mode when in 'sum' formulation. Please change to 'max'")
			return
		else if FISTA:
			print("Cannot run quasi-newton with FISTA")
			return
		else if BB:
			print("Cannot run quasi-newton with BB")
			return

	LIL2 = setOpts('LIL2', 0)
	if LIL2.isEmpty:
		LIL2 = 0

	if LIL2:
		if LIL2.lower().find('row'):
			LIL2 = 'rows'
		else if LIL.lower().find('col'):
			LIL2 = 'cols'
		else:
			print("unrecognized option for LIL2: should be row or column or 0")
			return

	projNuclear()
	if maxProject:
		project = lambda L, S, *args : projectMax(LIL2, tau, lambda_S, SVDopts, L, S)
	else if sumProject:
		if LIL2 > 0:
			print("error with opts[sum] = true, need opts[LIL2] =0")
			return
		project = lambda L, S, *args : projectSum(tau, lambda_S, L, S)
	else if Lagranian:
		project = lambda L, S, *args : projectMax(LIL2, lamda, lamda*lambda_S, SVDopts, L, S, *args)

	L = setOpts('L0', np.zeros(n1, n2))
	S = setOpts('S0', np.zeros(n1, n2))

	if not list(opts.keys()):
		print("warning, found extra guys in opts")
		print(opts)
		print("Found unprocessed options in opts")
		return

	stepsize = 1/Lip
	errHist = np.zeros(maxIts, 2 + not errFcn)
	Grad = 0

	if FISTA or BB or QUASINEWTON:
		L_old = L
		S_old = S

	L_fista = L
	S_fista = S
	BREAK = false
	kk = 0
	timeRef = time.time()

	for k in range(0, maxIts):
		R = A(L_fista + S_fista) - AY
		Grad_old = Grad
		Grad = At(R)

		objL = math.inf
		if QUASINEWTON:
			if S_L_S:
				dL = L - L_old
				S_old = S 
				*unused, S_temp = project([], S - stepsizeQN*(Grad + dL), stepsizeQN)

				dS = S_temp - S_old
				L_old = L
				L, *unused, rnk, objL = project(L - stepsizeQN*(Grad + dL), stepsizeQN)

				dL = L - L_old
				*unused, S = project([], S - stepsizeQN*(Grad + dL), stepsizeQN)
			else:
				dS = S - S_old
				L_old = L
				L, *unused, rnk, objL = project(L - stepsizeQN*(Grad+dS), [], stepsizeQN)
				dL = L - L_old
				S_old = S
				*unused, S = project([], S - stepsizeQN*(Grad+dL), stepsizeQN)
		else:
			if BB and k > 1:
				stepsizeL, stepsizeS = compute_BB_stepsize(Grad, Grad_old, L, L_old, S, S_old, BB_split, BB_type)
				if math.isnan(stepsizeL) or math.isnan(stepsizeS):
					print("Warning: no BB stepsize possible since iterates have not changed!\n")
					stepsizeL = stepsize 
					stepsizeS = stepsize
			else:
				stepsizeL = stepsize
				stepsizeS = stepsize

			if FISTA or BB:
				L_old = L
				S_old = S 

			L = L_fista - stepsizeL*Grad
			S = S_fista - stepsizeS*Grad
			#need to check for isnan(L(:)) or isnan(S(:))
			L, S, rank, objL = project(L, S, stepsizeL, stepsizeS)

		DO_RESTART = false
		if FISTA:
			if k>1 and restart > 0 and not np.isinf(restart) and not np.mod(kk, restart):
				kk=0
				DO_RESTART = true
			else if restart == float("-inf") && kk > 5:
				if (errHist(k-1, 2) - errHist(k-5, 2)) > math.exp(-8) * math.abs(errHist(k-5, 2)):
					DO_RESTART = true
					kk=0

			L_fista = L + kk/(kk+3) * (L - L_old)
			S_fista = S + kk/(kk+3) * (S - S_old)
			kk = kk + 1
		else:
			L_fista = L
			S_fista = S


		R = A(L+S) - AY

		res = np.linalg.norm(R[:])
		errHist(k, 1) = res
		errHist(k, 2) = 1/2*(res**2)
		if Lagranian:
			errHist(k, 2) = errHist(k, 2) + lamda*objL

			if LIL2 > 0:
				if LIL2 == 'rows':
					errHist[k, 2] = errHist[k, 2] + lamda*lambda_S*np.sum(np.sqrt(np.sum(np.power(S, 2), 2)))
				else:
					errHist[k, 2] = errHist[k, 2] + lamda*lambda_S*np.sum(np.sqrt(np.sum(np.power(S, 1), 2)))
			else:
				errHist[k, 2] = errHist[k, 2] + lamda*lambda_S*np.linalg.norm(S[:], 1)

		if k > 1 and math.abs(np.diff(errHist[k+1:k, 1]))/res < tol:
			BREAK = true

		PRINT = not np.mod(k, printEvery) or BREAK or DO_RESTART

		if PRINT:
			print("Iter %4d, rel. residual %.2e, objective %.2e", k, res/normAY, errHist[k, 2] -trueObj)

		if not errFcn:
			err = errFcn[L, S]
			errHist[k, 3]
			if PRINT:
				print("err %.2e", err)

		if not rnk and PRINT:
			print("rank(L) %3d", rnk)

		if PRINT:
			print("sparsity(S) %5.1f", 100*np.count_nonzero(S)/np.size(S))

		if displayTime and PRINT:
			tm = time.time() - timeRef
			print("time %.1f s", tm)

		if DO_RESTART:
			print("[restarted FISTA]")

		if PRINT:
			print("\n")

		if BREAK:
			print("Reached stopping criteria (Based on change in residual)\n")

	if BREAK:
		errHist = errHist[1:k, :]
	else:
		print("Reached maximum number of allowed iterations\n")

	return L, S, errHist

def projectMax(LIL2, tau, lambda_S, SVDopts, L, S, stepsize, stepsizeS):
	if len(locals()) >= 7 and not stepsize:
		tauL = -(math.abs(tau*stepsize))
		if len(locals()) < 8 or not stepsizeS:
			stepsizeS = stepsize
		tauS = -(math.abs(lambda_S*stepsizeS))
	else:
		tauL = math.abs(tau)
		tauS = math.abs(tau/lambda_S)

	if L:
		L, rnk, nuclearNorm = projNuclear(tauL, L, SVDopts)
		if tauL > 0:
			nuclearNorm = 0
	else:
		rnk=[]
		nuclearNorm = 0

	if S:
		if tauS > 0:
			if LIL2 > 0:
				projS = project_l1(tauS)
			else if LIL2 == 'rows':
				projS = project_l1l2(tauS, true)
			else if LIL2 == 'cols':
				projS = project_l1l2(tauS, false)
			else:
				print("bad value for LIL2: should be [], 'rows', or 'cols'")

			S = projS(S)
	else:
		if LIL2 > 0:
			S = np.multiply(np.sign(S), math.max(0, math.abs(S) - math.abs(tauS)))
		else if LIL2 == 'rows':
			projS = prox_l1l2(math.abs(tauS))
			S = projS(S, 1)
		else if LIL2 == 'cols':
			projS = prox_l1l2(math.abs(tauS))
			S = projS(numpy.matrix.traspose(S), 1)
		else:
			error("Bad value for LIL2: should be [], 'rows', or 'cols'")

	return L, S, rnk, nuclearNorm

#Global variables for projNuclear() function
oldRank = []
Vold = []
iteration = 0

def projNuclear(tau, X, SVDopts):
	global oldRank
	global Vold
	global iteration

	if len(locals()) == 0:
		oldRank = []
		Vold = []
		iteration = 0

	if not oldRank:
		rEst =10
	else:
		rEst = oldRank + 2

	if not iteration:
		iteration = 0

	iteration = iteration + 1
	n1, n2 = X.shape
	minN = math.min(n1, n2)

	if iteration == 1:
		rankMax = int(round(minN/4))
	else if iteration ==2:
		rankMax = int(round(minN/2))
	else:
		rankMax = minN

	style = SVDopts[SVDstyle]

	if tau ==0:
		X = 0*X
		return

	if style == 1:
		U, S, V = np.linalg.svd(X, 'econ')
		s = np.diag(S)

		if tau < 0:
			s = math.max(0, 0-s.math.abs(tau))
		else:
			s = project_simplex(tau, s)

		tt = s > 0
		rEst = np.count_nonzero(tt)
		U = U[:,tt]
		S = np.diag(s(tt))
		V = V[:,tt]
		nrm = np.sum(s(tt))
	else if style == 2 or style == 3 or style == 4:
		if style ==2:
			opts = {'tol' : math.exp(-4)}
			if rankMax == 1:
				opts[tol] = math.min(opts[tol], math.exp(-6))
			svdFcn = lambda X, rEst : numpy.sparse.linalg.svds(X, rEst, 'L', opts)
		else if style == 3:
			opts = {'tol' : math.exp(-4), 'eta', np.spacing(1)}
			opts[delta] = 10*opts[eta]
			if rankMax == 1:
				opts[tol] = math.min(opts[tol], math.exp(-6))
			svdFcn = lambda X, rEst : lansvd(X, rEst, 'L', opts) #PROBABLY NEED TO IMPLEMENT LANSVD
		else if style == 4:
			opts = []
			if 'nPower' in SVDopts and SVDopts[nPower]:
				nPower = SVDopts[nPower]
			else:
				nPower = 2

			if 'offset' in SVDopts and SVDopts[offset]:
				offset = SVDopts[offset]
			else:
				offset = 5

			if  'warmstart' in SVDopts and SVDopts[warmstart] == true and Vold:
				opts = {'warmstart' : Vold}

			ell = lambda r : math.min(r+offset, n1, n2)
			svdFcn = lambda X, rEst : randomizedSVD(X, rEst, ell(rEst), nPower, [], opts) #IMPLEMENT RANDOMIZED SVD TOO

		ok = false
		while not ok:
			rEst = math.min(rEst, rankMax)
			U, S, V = svdFcn(X, rEst)
			s = np.diag(S)

			if tau < 0:
				lamda = math.abs(tau)
			else:
				lamda = findTau(s, tau)

			ok = (math.min(s) < lamda) or (rEst == rankMax)
			if ok:
				break
			rEst = 2*rEst

		rEst = math.min(len(numpy.where(s>lamda)), rankMax)
		S = np.diag(s(1:rEst)-lamda)
		U = U[:,1:rEst]
		V = V[:,1:rEst]
		nrm = np.sum(s(1:rEst) - lamda)
	else:
		print("bad value for SVDstyle")
		return

	if not U:
		X = 0*X
	else:
		X = U*S*V

	oldRank = U.shape[1]
	if 'warmstart' in SVDopts and SVDopts[warmstart] == true:
		Vold = V

	return X, rEst, nrm

def project_simplex(q, x):
	x = np.multiply(x, (x>0))

	if np.sum(x) <= q:
		return

	if q==0:
		x=0*x
		return
	s = np.sort(x) #look into descending sort
	if q < np.spacing(s(1)):
		print("Input is scaled so large compared to q that accurate computations are difficult")

	cs = np.divide((np.cumsum(s) - 1), (np.transpose(1:len(s)))) # CHECK THIS
	ndx = np.count_nonzero(s>cs)
	x = math.max(x - cs(ndx), 0)

	return x

def findTau(s, lamda):

	if np.all(s==0) or lamda = 0:
		tau=0
		return tau

	if size(s) > len(s): #CHECK THIS CHECK
		print("s should be a vector, not a matrix")
		return

	if sorted(np.flipud(s)) not flipud(s):
		s = np.sort(s)

	if (s<0):
		print("s should be non-negative")

	cs = np.divide((np.cumsum(s) - math.abs(lamda)), (1:len(s))) #CHECK THIS
	ndx = np.count_nonzero(s > cs)
	tau = math.max(0, cs(ndx))

	return tau

def projectSum(tau, lambda_S, L, S):
	m, n = L.shape
	U, Sigma, V = np.linalg.svd(L, 'econ')
	s = np.diag(Sigma)
	wts = [np.ones(len(s), 1); lambda_S*np.ones(m*n, 1)]
	proj = project_l1(tau, wts)
	sS = proj([s;vec(S)])
	sProj = sS(1:len(s))
	S = np.reshape(sS[len(s)+1: end], (m, n))
	L = U*np.diag(sProj)*V
	rnk = np.count_nonzero(sProj)
	nuclearNorm = np.sum(sProj)

	return L, S, rnk, nuclearNorm

def compute_BB_stepsize(Grad, Grad_old, L, L_old, S, S_old, BB_split, BB_type):
	if not BB_split:
		yk = np.subtract(Grad[:], Grad_old[:])
		yk = [yk;yk]
		sk = [np.subtract(L[:], L_old[:]); np.subtract(S[:], S_old[:])]
		if BB_type == 1:
			stepsize = np.linalg.norm(sk**2)/(np.transpose(sk) * yk)
		else if BB_type == 2:
			stepsize = np.transpose(sk) * yk / (np.linalg.norm()**2)

		stepsizeL = stepsize
		stepsizeS = stepsize

	else if BB_split:
		yk = np.subtract(Grad[:], Grad_old[:])
		skL = np.subtract(L[:], L_old[:])
		skS = np.subtract(S[:], S_old[:])
		if BB_type == 1:
			stepsizeL = (np.linalg.norm(skL)**2)/(np.transpose(skL)*yk)
			stepsizeS = (np.linalg.norm(skS)**2)/(np.transpose(skS)*yk)
		else if BB_type == 2:
			stepsizeL = (np.transpose(skL)*yk)/(np.linalg.norm(yk)**2)
			stepsizeS = (np.transpose(skS)*yk)/(np.linalg.norm(yk)**2)

	return stepsizeL, stepsizeS

def project_l1(q, d):
#PROJECT_L1   Projection onto the scaled 1-norm ball.
#    OP = PROJECT_L1( Q ) returns an operator implementing the 
#    indicator function for the 1-norm ball of radius q,
#    { X | norm( X, 1 ) <= q }. Q is optional; if omitted,
#    Q=1 is assumed. But if Q is supplied, it must be a positive
#    real scalar.
#
#    OP = PROJECT_L1( Q, D ) uses a scaled 1-norm ball of radius q,
#    { X | norm( D.*X, 1 ) <= 1 }. D should be the same size as X
#    and non-negative (some zero entries are OK).

# Note: theoretically, this can be done in O(n)
#   but in practice, worst-case O(n) median sorts are slow
#   and instead average-case O(n) median sorts are used.
#   But in matlab, the median is no faster than sort
#   (the sort is probably quicksort, O(n log n) expected, with
#    good constants, but O(n^2) worst-case).
#   So, we use the naive implementation with the sort, since
#   that is, in practice, the fastest.

	if len(locals()) == 0:
		q = 1
	else if not q.isnumeric() or not np.isreal(q) or size(q) != 1 or q<=0:
		print("[RPCA_c: projl1] Argument must be positive")
		return

	if len(locals()) < 2 or not d or size(d) == 1:
		if len(locals()) >= 2 and d:
			if d == 0:
				print("[RPCA_c: projl1] if d==0 in proj_l1, the set is just {0}, so use proj_0")
				return
			else if d < 0:
				print("[RPCA_c: projl1] Require d >= 0")
			q=q/d
		op = lambda *args : proj_l1_q(q, *args)
	else:
		if np.nonzero(d<0):
			print("[RPCA_c] all entries of d must be non-negative")
		op = lambda *args : proj_l1_q_d(q, d, *args)

	def proj_l1_q(q, x, *args):
		myReshape = lambda x : x
		if x.shape(1) > 1:
			if np.ndim(x) > 2:
				print("[RPCA_c : proj_l1_q] You must modify this code to deal with tensors")
				return
			myReshape = lambda y : np.reshape(y, x.shape(0), x.shape(1))
			x = x[:]

		s = np.sort(math.abs(np.zeros(x)))
		cs = np.cumsum(s)
		ndx = np.nonzero(np.multiply(cs-np.transpose(1:size(s)), [s:1:end; 0] >= q+2*np.spacing(q-1))) #CHECK THIS
		if not ndx:
			thresh = ( cs(ndx) - q) / ndx
			x = np.multiply(x, 1 - (np.divide(thresh, math.max(math.abs(x), thresh))))
		x = myReshape(x)
		return x

	def proj_l1_q_d(q, d, x, *args):
		myReshape = lambda x : x
		if x.shape(1) > 1:
			if np.ndim(x) > 2:
				print("[RPCA_c : proj_l1_q] You must modify this code to deal with tensors")
				return
			myReshape = lambda y : np.reshape(y, x.shape(0), x.shape(1))
			x = x(:)

		goodInd, j, xOverD = np.nonzero(np.divide(x, d))
		lambdas, srt = np.sort(math.abs(xOverD)) #LOOK INTO DESCENDING SORT
		s = math.abs(np.multiply(x(goodInd), d(goodInd)))
		s = s(srt)
		dd = np.power(d(goodInd), 2)
		dd = dd(srt)
		cs = np.cumsum(s)
		cd = np.cumsum(dd)
		ndx = np.nonzero(np.multiply(cs-lambdas, cd>=q+2*np.spacing(q-1), 1, 'first')) #CHECK THIS
		if not ndx:
			ndx = ndx-1
			lamda = (cs(ndx)-q)/cd(ndx)
			x = np.multiply(np.sign(x), max(0, math.abs(x) - lamda*d))
		x = myReshape(x)
		return x


def project_l1l2(q = 1, rowNorms = true):
	if not q.isnumeric() or not np.isreal(q) or np.nonzero(q<=0) or size(q) > 1:
		print("[RPCA_c: proj_l1l2] Argument must be positive and a scalar")

	if rowNorms:
		op = lambda x, *args : prox_f_rows(q, x)
	else:
		op = lambda x, *args : prox_f_cols(q, x)


	def prox_f_rows(tau, X):
		nrms = math.sqrt(np.sum(np.power(X, 2), 1))
		s = sort(nrms)
		cs = np.cumsum(s)

		ndx = np.nonzero(np.multiply(cs - np.transpose(1:size(s)), [s(1:end); 0] >= tau+2*np.spacing(tau-1)))

		if not ndx:
			thresh = (cs(ndx) - tau) / ndx

			d = math.max(0, np.divide(1 - thresh, nrms))
			m = X.shape(0)
			X = scipy.sparse.spdiags(d, 0, n, n)*X

		return X

	def prox_f_cols(tau, X):
		nrms = math.sqrt(np.sum(np.power(X, 2), 0))
		s = np.sort(nrms)
		cs = np.cumsum(s)

		ndx = np.nonzero(np.multiply(cs - np.transpose(1:size(s)), [s(1:end); 0] >= tau+2*np.spacing(tau-1)))
		if not ndx:
			thresh = (cs(ndx) - tau)/ ndx
			d = math.max(0, 1-np.divide(thresh, nrms))
			n = X.shape(1)
			X = X*scipy.sparse.spdiags(d, 0, n, n)

		return X

def prox_l1l2(q = 1):
	if not q.isnumeric() or not np.isreal(q) or np.nonzero(q<=0):
		print("[RPCA_c : prox_l1l2] Argument must be positive")

	op = lambda x, t : prox_f(q, x, t)

	def prox_f(q, x, t):
		if len(locals()) < 3:
			print ("[RPCA_c : prox_f] Not enough arguments")

		V = math.sqrt(np.sum(np.power(x, 2), 1))
		s = 1 - np.divide(1, math.max(np.divide(v, np.multiply(t, q)), 1))
		m = len(s)
		x = scipy.sparse.spdiags(x, 0, m, m)*x

		return x

	return op
