import numpy as np 
import os
import scipy
import math

def lansvd(A,  m = A.shape[0], n = A.shape[1], K = math.min(math.min(m, n), 6), SIGMA='L', OPTS=[]):
	#LANSVD  Compute a few singular values and singular vectors.
#   LANSVD computes singular triplets (u,v,sigma) such that
#   A*u = sigma*v and  A'*v = sigma*u. Only a few singular values 
#   and singular vectors are computed  using the Lanczos 
#   bidiagonalization algorithm with partial reorthogonalization (BPRO). 
#
#   S = LANSVD(A) 
#   S = LANSVD('Afun','Atransfun',M,N)  
#
#   The first input argument is either a  matrix or a
#   string containing the name of an M-file which applies a linear
#   operator to the columns of a given matrix.  In the latter case,
#   the second input must be the name of an M-file which applies the
#   transpose of the same operator to the columns of a given matrix,  
#   and the third and fourth arguments must be M and N, the dimensions 
#   of the problem.
#
#   [U,S,V] = LANSVD(A,K,'L',...) computes the K largest singular values.
#
#   [U,S,V] = LANSVD(A,K,'S',...) computes the K smallest singular values.
#
#   The full calling sequence is
#
#   [U,S,V] = LANSVD(A,K,SIGMA,OPTIONS) 
#   [U,S,V] = LANSVD('Afun','Atransfun',M,N,K,SIGMA,OPTIONS)
#
#   where K is the number of singular values desired and 
#   SIGMA is 'L' or 'S'.
#
#   The OPTIONS structure specifies certain parameters in the algorithm.
#    Field name      Parameter                              Default
#   
#    OPTIONS.tol     Convergence tolerance                  16*eps
#    OPTIONS.lanmax  Dimension of the Lanczos basis.
#    OPTIONS.p0      Starting vector for the Lanczos        rand(n,1)-0.5
#                    iteration.
#    OPTIONS.delta   Level of orthogonality among the       sqrt(eps/K)
#                    Lanczos vectors.
#    OPTIONS.eta     Level of orthogonality after           10*eps^(3/4)
#                    reorthogonalization. 
#    OPTIONS.cgs     reorthogonalization method used        0
#                    '0' : iterated modified Gram-Schmidt 
#                    '1' : iterated classical Gram-Schmidt
#    OPTIONS.elr     If equal to 1 then extended local      1
#                    reorthogonalization is enforced. 
#
#   See also LANBPRO, SVDS, SVD

# References: 
# R.M. Larsen, Ph.D. Thesis, Aarhus University, 1998.
#
# B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
# Prentice-Hall, Englewood Cliffs, NJ, 1980.
#
# H. D. Simon, ``The Lanczos algorithm with partial reorthogonalization'',
# Math. Comp. 42 (1984), no. 165, 115--142.

# Rasmus Munk Larsen, DAIMI, 1998

# Python Implementation: Matt Dioso, Seattle University Department of Electrical and Computer Engineering 2018
# NOTE: Purposely omitting the Atrans option. don't think it applies in Python implementation
# ref: https://github.com/areslp/matlab/blob/master/PROPACK/lansvd.m

##################### Parse and check input arguments. ######################

	if not isinstance(A, str):
		if not np.isreal(A):
			print("[lansvd] A must be real")
			return

	if m < 0 or n < 0 or K < 0:
		print("[lansvd] m, n, and K must be positive integers")
		return


	if math.min(n, m) < 1 or K < 1:
		U = np.eye(m, K)
		S = np.zeroes(K, K)
		V = np.eye(n, K)
		bnd = np.zeroes(K, 1)

		return U, S, V

	else if math.min(n, m) == 1 and K > 0:
		U, S, V = np.linalg.svd(A) #translated from [U,S,V] = svd(full(A))		
		bnd = 0

		return U, S, V

	if isinstance(A, num):
		if np.count_nonzero(A) == 0:
			U = np.eye(m, K)
			S = np.zeroes(K, K)
			V = np.eye(n, K)
			bnd = np.zeros(K, 1)

			return U, S, V

	lanmax = math.min(m, n)
	tol = 16 * np.spacing(1)
	p = np.random.rand(m, 1) - 0.5

	c = list(opts.keys())
	for i in range(0, len(c)):
		if "p0" in c:
			p = opts.get("p0")
			p = p[:]
		if "tol" in c:
			tol = opts.get("tol")
		if "lanmax" in c:
			lanmax = opts.get("lanmax")

	tol = math.max(tol, np.spacing(1))
	lanmax = math.min(lanmax, math.min(m, n))
	if p.shape(0) != m:
		print("[lansvd] p0 must be a vector of length m")
		return

	lanmax = math.min(lanmax, math.min(m, n))
	if K > lanmax:
		print("[lansvd] K must satisfy K <= LANMAX <= MIN(M, N)")

	if SIGMA == 'S':
		if isinstance(A, str):
			print("[lansvd] Shift-and-invert works only when the atrix A is given explicitly")
		else:
			if scipy.sparse.issparse(A):
				pmmd = scipy.sparse.linalg.splu(A, 'COLAMD')
				AA = A[:,pmmd] #CHECK THIS
			else:
				AA = A #ALSO CHECK THIS

			if m>=n:
				if scipy.sparse.issparse(AA):
					AR = scipy.linalg.qr(AA)
					ARt = AR.conj().transpose()
					p = ARt / (AA.conj().transpose().dot(p))
				else:
					AQ, AR = scipy.linalg.qr(AA)
					ARt = AR.conj().transpose()
					p = AQ.conj.transpose().dot(p)
			else:
				print("Sorry, shift-and-invert for m<n not implemented yet")
				AR = scipy.linalg.qr(AA.conj().transpose())
				ARt = AR.conj().transpose()
			condR = np.linalg.cond(AR)
			if condR > 1/np.spacing(1):
				print("[lansvd] A is rank deficient or too ill-conditioned to do shift-and-invert")
				return

	ksave = K
	neig = 0
	nrestart = -1
	j = math.min(k+math.max(8, K) + 1, lanmax)
	U = []
	V = []
	B = []
	anorm = []
	work = np.zeros(2, 2)
	
	while neig < K:
		if not isinstance(A, str):
			U, B, V, p, ierr, w = lanbpro(A, j, p, opts, U, B, V, anorm) #IMPLEMENT LANBPRO

		work = work +w

		if ierr < 0:
			j = -ierr

		resnrm = np.linalg.norm(p)

		S, bot = bdsqr(np,diag(B), [np.diag(B, -1); resnrm]) #IMPLEMENT BDSQR

		anorm = S[0]

		bnd = resnrm * math.abs(bot)

		bnd = refinebounds(S**2, bnd, n*np.spacing(1)*anorm)

		i = 0
		neig = 0
		while i <= math.min(j, k):
			if (bnd[i] <= tol*math.abs(S[i])):
				neig = neig + 1
				i = i + 1
			else:
				i = math.min(j, k) + 1

		if ierr < 0:
			if j < k:
				print("[lansvd] Warning: Invariant subspace of dimension %d found", j-1)
			j = j - 1
			break

		if j >= lanmax:
			if j >= math.min(m,n):
				neig = ksave
				break

			if neig < ksave:
				print("[lansvd] Warning maximum dimension of Krylob subspace exceeded prior to convergence")

			break
		else:
			j = math.max(1.5*j, j+10)
		
		if neig > 0:
			j = j+math.min(100, math.max(2, 0.5*(K-neig)*j/(neig+1)))
		else:
			j=math.max(1.5*j, j+10)

		j = math.ceil(math.min(j+1, lanmax))	
		nrestart = nrestart + 1

	K = math.min(ksave, j)

	j=B.shape[1]

	P, S, Q = np.linalg.svd([B;np.zeros(1, j-1)])

	S = np.diag(S)

	if Q.shape[1] != K:
		Q = Q[:, 0:K]
		P = P[:, 0:K]

	if resnrm != 0:
		U = U.dot(P[0:j, :]) + (p/resnrm).dot(P[j+1, :])
	else:
		U = U.dot(P[0:j, :])

	V = V.dot(Q)

	for i in range(0, K):
		nq = np.linalg.norm(V[:, i])
		if np.isfinite(nq) and nq != 0 and nq != 1:
			V[:,i] = U[:, i]/nq

		nq = np.linalg.norm(U[:, i])
		if np.isfinite(nq) and nq != 0 and nq != 1:
			U[:, i] = U[:, i]/nq

	S = S[0:K]
	bnd = bnd[0:K]

	if sigma == 'S':
		S= np.sort(-1/S)
		S = -S
		bnd = bnd[S.size]
		if scipy.sparse.issparse(AA):
			U = AA.dot(AR/U[:,p])
			V[pmmd,:] = V[:,p]
		else:
			U = AQ[:, 1:math.min(m,n)].dot(U[:, p])
			V = V[:,p]

	return U, S, V, bnd, j
