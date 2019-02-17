import os
import numpy as np 
import scipy as sp 
import math

LANBPRO_TRUTH = 0
if LANBPRO_TRUTH == 1:
	MU = 0
	NU = 0
	MUTRUE = 0
	NUTRUE = 0
	MU_AFTER = 0
	NU_AFTER = 0
	MUTRUE_AFTER = 0
	NUTRUE_AFTER = 0

def lanbpro(A, K, RO, Opts = [], U = [], B_k = [], V = []):
#	LANBPRO Lanczos bidiagonalization with partial reorthogonalization.
#   LANBPRO computes the Lanczos bidiagonalization of a real 
#   matrix using the  with partial reorthogonalization. 
#
#   [U_k,B_k,V_k,R,ierr,work] = LANBPRO(A,K,R0,OPTIONS,U_old,B_old,V_old) 
#   [U_k,B_k,V_k,R,ierr,work] = LANBPRO('Afun','Atransfun',M,N,K,R0, ...
#                                       OPTIONS,U_old,B_old,V_old) 
#
#   Computes K steps of the Lanczos bidiagonalization algorithm with partial 
#   reorthogonalization (BPRO) with M-by-1 starting vector R0, producing a 
#   lower bidiagonal K-by-K matrix B_k, an N-by-K matrix V_k, an M-by-K 
#   matrix U_k and an M-by-1 vector R such that
#        A*V_k = U_k*B_k + R
#   Partial reorthogonalization is used to keep the columns of V_K and U_k
#   semiorthogonal:
#         MAX(DIAG((EYE(K) - V_K'*V_K))) <= OPTIONS.delta 
#   and 
#         MAX(DIAG((EYE(K) - U_K'*U_K))) <= OPTIONS.delta.
#
#   B_k = LANBPRO(...) returns the bidiagonal matrix only.
#
#   The first input argument is either a real matrix, or a string
#   containing the name of an M-file which applies a linear operator 
#   to the columns of a given matrix. In the latter case, the second 
#   input must be the name of an M-file which applies the transpose of 
#   the same linear operator to the columns of a given matrix,  
#   and the third and fourth arguments must be M and N, the dimensions 
#   of then problem.
#
#   The OPTIONS structure is used to control the reorthogonalization:
#     OPTIONS.delta:  Desired level of orthogonality 
#                     (default = sqrt(eps/K)).
#     OPTIONS.eta  :  Level of orthogonality after reorthogonalization 
#                     (default = eps^(3/4)/sqrt(K)).
#     OPTIONS.cgs  :  Flag for switching between different reorthogonalization
#                     algorithms:
#                      0 = iterated modified Gram-Schmidt  (default)
#                      1 = iterated classical Gram-Schmidt 
#     OPTIONS.elr  :  If OPTIONS.elr = 1 (default) then extended local
#                     reorthogonalization is enforced.
#     OPTIONS.onesided
#                  :  If OPTIONS.onesided = 0 (default) then both the left
#                     (U) and right (V) Lanczos vectors are kept 
#                     semiorthogonal. 
#                     OPTIONS.onesided = 1 then only the columns of U are
#                     are reorthogonalized.
#                     OPTIONS.onesided = -1 then only the columns of V are
#                     are reorthogonalized.
#     OPTIONS.waitbar
#                  :  The progress of the algorithm is display graphically.
#
#   If both R0, U_old, B_old, and V_old are provided, they must
#   contain a partial Lanczos bidiagonalization of A on the form
#
#        A V_old = U_old B_old + R0 .  
#
#   In this case the factorization is extended to dimension K x K by
#   continuing the Lanczos bidiagonalization algorithm with R0 as a 
#   starting vector.
#
#   The output array work contains information about the work used in
#   reorthogonalizing the u- and v-vectors.
#      work = [ RU  PU ]
#             [ RV  PV ] 
#   where
#      RU = Number of reorthogonalizations of U.
#      PU = Number of inner products used in reorthogonalizing U.
#      RV = Number of reorthogonalizations of V.
#      PV = Number of inner products used in reorthogonalizing V.

# References: 
# R.M. Larsen, Ph.D. Thesis, Aarhus University, 1998.
#
# G. H. Golub & C. F. Van Loan, "Matrix Computations",
# 3. Ed., Johns Hopkins, 1996.  Section 9.3.4.
#
# B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
# Prentice-Hall, Englewood Cliffs, NJ, 1980.
#
# H. D. Simon, ``The Lanczos algorithm with partial reorthogonalization'',
# Math. Comp. 42 (1984), no. 165, 115--142.
#

# Rasmus Munk Larsen, DAIMI, 1998.
# Python Implementation: Matt Dioso, Seattle University Department of Electrical and Computer Engineering 2018
# NOTE: Purposely omitting the Atrans option. don't think it applies in Python implementation
# Ref: https://github.com/areslp/matlab/blob/master/PROPACK/lanbpro.m

	global LANBPRO_TRUTH
	global MU
	global NU 
	global MUTRUE
	global NUTRUE
	global MU_AFTER
	global NU_AFTER
	global MUTRUE_AFTER
	global NUTRUE_AFTER

	if len(locals()) < 1 or len(locals()) < 2:
		print("[LANBPRO] Not enough input arguments")
		return

	narg = len(locals())

	if not np.isreal(A):
		print("[LANPRO] A must be real")
		return
	m, n = A.shape

	if narg >= 3:
		p = RO
	else:
		p = np.random.rand(m, 1) - 0.5

	anorm = []

	if math.min(m, n) == 0:
		U = []
		B_k = []
		V = []
		p = []
		ierr = 0
		work = np.zeros(2,2)
		return U, B_k, V, p, ierr, work
	elif math.min(m, n) == 1:
		U = 1
		B_k = A
		V = 1
		p = 0
		ierr = 0
		work = np.zeros(2, 2)
		return U, B_k, V, p, ierr, work

	m2 = 3/2
	n2 = 3/2
	delta = math.sqrt(np.spacing(1)/K)
	eta = np.spacing(1) ** (3/4)/math.sqrt(K)
	cgs = 0
	elr = 2

	gamma = 1/math.sqrt(2)
	onesided = 0
	t = 0
	waitb = 0

	if not opts and isinstance(opts, dict):
		c = opts.keys()
		if "delta" in c:
			delta = opts.get("delta")
		if "eta" in c:
			eta = opts.get("eta")
		if "cgs" in c:
			cgs = opts.get("cgs")
		if "elr" in c:
			elr = opts.get("elr")
		if "gamma" in c:
			gamma = opts.get("gamma")
		if "onesided" in c:
			onesided = opts.get("onsided")
		if "waitbar" in c:
			waitb = 1

	#if waitb:
		#wait

	if not anorm:
		anorm = []
		est_anorm = 1
	else:
		est_anorm = 0

	FUDGE = 1.01

	npu = 0
	npv = 0
	ierr = 0
	p = p[:]

	if not U:
		V = np.zeros(n, K)
		U = np.zeros(m, K)
		beta = np.zeros(K+1, 1)
		alpha = np.zeros(K, 1)
		beta[0] = np.linalg.norm(p)
		nu = np.zeros(K, 1)
		mu = np.zeros(K+1, 1)
		mu[0] = 1
		nu[0] = 1

		numax = np.zeros(K, 1)
		mumax = np.zeros(K, 1)
		force_reorth = 0
		nreorthu = 0
		nreorthv = 0
		j0 = 1
	else:
		j = U.shape[1]

		U = np.append(U, np.zeros(m, K-j))
		V = np.append(V, np.zeros(n, K-j))
		alpha = np.zeros(K+1, 1)
		beta = np.zeros(K+1, 1)
		alpha[0:j] = np.diag(B_k)
		if j > 1:
			beta[1:j] = np.diag(B_k, -1)
		beta[j+1] = np.linalg.norm(p)

		if j<K and beta[j+1] * delta < anorm * np.spacing(1):
			fro = 1
			ierr = j

		int = [0:j].conj().transpose
		p, beta[j+1], rr = reorth(U, p, beta[j+1], int, gamma, cgs) #IMPLEMENT REORTH
		npu = rr*j
		nreorthu = 1
		force_reorth = 1

		if est_anorm:
			anorm = FUDGE * math.sqrt(np.linalg.norm(B_k.conj().transpose * B_k, 1))

		mu = m2*np.spacing(1) * np.ones(K+1, 1)
		nu = np.zeros(K, 1)
		numax = np.zeros(K, 1)
		mumax = np.zeros(K, 1)
		force_reorth = 1
		nreorthu = 0
		nreorthv = 0
		j0 = j+1

	At = A.conj().transpose()

	if delta == 0:
		fro = 1
	else:
		fro = 0

	if LANBPRO_TRUTH == 1:
		MUTRUE = np.zeros(K, K)
		NUTRUE = np.zeros(K-1, K-1)
		MU = np.zeros(K, K)
		NU = np.zeros(K-1, K-1)

		MUTRUE_AFTER = np.zeros(K, K)
		NUTRUE_AFTER = np.zeros(K-1, K-1)
		MU_AFTER = np.zeros(K, K)
		NU_AFTER = np.zeros(K-1, K-1)

	for j in range(j0, K):
		if beta[j] != 0:
			U[:, j] = p/beta[j]
		else:
			U[:, j] = p

		if j == 6:
			B = np.append(np.diag(alpha[0:j-1])+np.diag(beta[1:j-1], -1), np.append(np.zeros(1, j-1), beta[j]))
			anorm = FUDGE * np.linalg.norm(B)
			est_anorm = 0

		if j == 1:
			r = At*U[:, 1]
			alpha[1] = np.linalg.norm(r)
			if est_anorm:
				anorm = FUDGE * alpha[1]

		else:
			r - At*U[:, j] - beta[j]*V[:, j-1]
			alpha[j] = np.linalg.norm(r)

			if alpha[j] < gamma*beta[j] and elr and not fro:
				normold = alpha[j]
				stop = 0
				while not stop:
					t = V[:, j-1].conj().transpose() * r
					r = r - V[:, j-1] * t
					alpha[j] = np.linalg.norm(r)
					if beta[j] != 0:
						beta[j] = beta[j] + t

					if alpha[j] >= gamma*normold:
						stop = 1
					else:
						normold = alpha[j]

			if est_anorm:
				if j==2:
					anorm = math.max(anorm, FUDGE*math.sqrt(alpha[0]**2 + beta[1]**2 + alpha[1]*beta[1]))
				else:
					anorm = math.max(anorm, FUDGE* math.sqrt(alpha(j-1)**2 + beta[j]**2+alpha[j-1]*beta[j-1] + alpha[j]*beta[j]))

			if not fro and alpha[j] != 0:
				nu = update_nu(nu, mu, j, alpha, beta, anorm)
				numax[j] = math.max(math.abs(nu[1:j-1]))

			if j>1 and LANBPRO_TRUTH:
				NU[0:j-1, j-1] = nu[0:j-1]
				NUTRUE[0:j-1, j-1] = V[:, 1:j-1].conj().transpose()*r/alpha[j]

			if elr > 0:
				nu[j-1] = n2*np.spacing(1)

			if onsided != -1 and (fro or numax[j] > delta or force_reorth) and alpha[j] != 0:
				if fro or eta == 0:
					int = [0:j-1].conj().transpose()
				elif force_reorth == 0:
					int = compute_int(nu, j-1, delta, eta, 0, 0, 0)

				r, alpha[j], rr = reorth(V, r, alpha[j], int, gamma, cgs)
				npv = npv + rr*len(int)
				nu[int] = n2*np.spacing(1)

				if force_reorth == 0:
					force_reorth = 1
				else:
					force_reorth = 0

				nreorthv = nreorthv + 1

		if alpha[j] < math.max(n,m)*anorm*np.spacing(1) and j < k:
			alpha[j] = 0
			bailout = 1
			for i in range(0, 2):
				r = np.random.rand(m, 1)-0.5
				r = At * r

				nrm = math.sqrt((r.conj().transpose()) * r)
				int = [1:j-1].conj().transpose()
				r, nrmnew, rr = reorth(V, r, nrm, int, gamma, cgs)
				npv = npv + rr*len(int[:])
				nreorthv = nreorthv + 1
				nu[int] = n2 * np.spacing(1)
				if nrmnew > 0:
					bailout = 0
					break

			if bailout:
				j=j-1
				ierr = -j
				break
			else:
				r = r/nrmnew
				force_reorth = 1
				if delta > 0:
					fro =0
		elif j<l and not fro and anorm*np.spacing(1) > delta*alpha[j]:
			fro =1
			ierr = j

		if J>1 and LANBPRO_TRUTH:
			NU_AFTER[0:j-1, j-1] = nu[0:j-1]
			NUTRUE_AFTER[1:j-1, j-1] = (V[:, 0:j-1].conj().transpose)*r/alpha[j]

		if alpha[j] != 0:
			V[:, j] = r/alpha[j]
		else:
			V[:,j] = r

		p = A * V[:, j] - alpha[j]*U[:,j]
		beta[J+1] = np.linalg.norm(p)

		if beta[j+1] < gamma*alpha[j] and elr and not fro:
			normold = beta[j+1]
			stop = 0
			while not stop:
				t = U[:, j].conj().transpose() * p
				p = p - U[:, j]*t
				beta[j+1] = np.linalg.norm(p)
				if alpha[j] != 0:
					alpha[j] = alpha[j] + t

				if beta[j+1] >= gamma*normold:
					stop = 1
				else:
					normold = beta[j+1]

		if est_anorm:
			if j==1:
				anorm = math.max(anorm, FUDGE*pythag(alpha[0], alpha[1]))
			else:
				anorm = math.max(anorm, FUDGE*math.sqrt(alpha[j] ** 2 + alpha[j]*beta[j]))

		if not fro and beta[j+1] != 0:
			mu = update_mu(mu, nu, j, alpha, beta, anorm)
			mumax[j] = math.max(math.abs(mu[1:j])) #i think this is getting the max of an array?

		if LANBPRO_TRUTH == 1:
			MU[0:j, j] = mu[0:j]
			MUTRUE[1:j, j] = U[:, 0:j].conj().transpose() * p/beta[j+1]

		if elr > 0:
			mu[j] = m2 * np.spacing(1)

		if onesided != 1 and (fro or mumax[j] > delta or force_reorth) and beta[j+1] != 0:
			if fro or eta ==1:
				int = [0:j].conj().transpose()
			elif force_reorth == 0:
				int = compute_int(mu, j, delta, eta, 0, 0, 0)
			else:
				int = [int; max[int]+1]

			p, beta[j+1], rr = reorth(U, p, beta[j+1], int, gamma, cgs)
			nppu = npu + rr*len(int)
			nreorthu = nreorthu + 1

			mu[int] = m2 * np.spacing(1)

			if force_reorth == 0:
				force_reorth =1
			else:
				force_reorth = 0


		if beta[j + 1] < math.max(m, n) * anorm * np.spacing(1) and j<k:
			beta[j+1] = 0
			bailout = 1
			for i in range(0, 2):
				p = np.random.rand(n, 1) - 0.5
				p = A * p
				nrm = math.sqrt((p.conj().transpose) * p)
				int = [0:j].conj().transpose()
				p, nrmnew, rr = reorth(U, p, nrm, int, gamma, cgs)
				npu = npu + rr*len(int[:])
				nreorthu = nreorthu + 1
				mu[int] = m2 * np.spacing(1)

				if nrmnew > 0:
					bailout = 0
					break

			if bailout: 
				ierr = -j
				break
			else:
				p = p/nrmnew
				force_reorth = 1
				if delta > 0:
					fro = 0
		elif j<k and not fro and anorm*np.spacing(1) > delta*beta[j+1]:
			fro = 1
			ierr = j


		if LANBPRO_TRUTH == 1:
			MU_AFTER[0:j, j] = mu[0:j]
			MUTRUE_AFTER[0:j, j] = (U[:, 0:j].conj().transpose())*p/beta[j+1]

	if j<K:
		j=K

	B_k = scipy.sparse.spdiags([alpha[1:K]; [beta[1:K]; 0]], [0, -1], K, K)

	if K != U.shape(1) or K != V.shape(1):
		U = U[:, 0:K]
		V = V[:, 0:K]

	work = [[nreorthu, npu]; [nreorthv, npv]]

	return U, B_k, V, p, ierr, work
	

