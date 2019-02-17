import numpy as np
import scipy as spy 
import os
import math

def reorth(Q, r, normr, index, alpha = 0.5, method = 0):
#REORTH   Reorthogonalize a vector using iterated Gram-Schmidt
#
#   [R_NEW,NORMR_NEW,NRE] = reorth(Q,R,NORMR,INDEX,ALPHA,METHOD)
#   reorthogonalizes R against the subset of columns of Q given by INDEX. 
#   If INDEX==[] then R is reorthogonalized all columns of Q.
#   If the result R_NEW has a small norm, i.e. if norm(R_NEW) < ALPHA*NORMR,
#   then a second reorthogonalization is performed. If the norm of R_NEW
#   is once more decreased by  more than a factor of ALPHA then R is 
#   numerically in span(Q(:,INDEX)) and a zero-vector is returned for R_NEW.
#
#   If method==0 then iterated modified Gram-Schmidt is used.
#   If method==1 then iterated classical Gram-Schmidt is used.
#
#   The default value for ALPHA is 0.5. 
#   NRE is the number of reorthogonalizations performed (1 or 2).

# References: 
#  Aake Bjorck, "Numerical Methods for Least Squares Problems",
#  SIAM, Philadelphia, 1996, pp. 68-69.
#
#  J.~W. Daniel, W.~B. Gragg, L. Kaufman and G.~W. Stewart, 
#  ``Reorthogonalization and Stable Algorithms Updating the
#  Gram-Schmidt QR Factorization'', Math. Comp.,  30 (1976), no.
#  136, pp. 772-795.
#
#  B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
#  Prentice-Hall, Englewood Cliffs, NJ, 1980. pp. 105-109

#  Rasmus Munk Larsen, DAIMI, 1998.

# Check input arguments.
# warning('PROPACK:NotUsingMex','Using slow matlab code for reorth.')

	n, k1 = np.shape(Q)
	if not normr:
		normr = math.sqrt((r.conj().transpose()) * R)

	if not index:
		k =k1
		index = [0:k].conj().transpose()
		simple =1
	else:
		k = len(index)
		if k == k1 and index[:]==[0:k].conj().transpose():
			simple = 1
		else:
			simple = 0

	if k==0 or n==0:
		return

	s = np.zeros(k, 1)

	normr_old = 0
	nre = 0

	while normr < alpha*normr_old or nre ==0:
		if method == 1:
			if simple:
				t = (Q.conj().transpose()) * r
				r = r - (Q*t)
			else:
				t = Q[:, index].conj().transpose() * r
				r = r - Q[:, index] * t
		else:
			for i in range(0, index):
				t = Q[:, i].conj().tranpose() * r
				r = r - Q[:, i] * t

		s = s + t
		normr_old = normr
		normr = math.sqrt(r.conj().transpose()) * r
		nre = nre + 1
		if nre > 4:
			r = np.zeros(n, 1)
			normr = 0
			return r, normr, nre, s
	return r, normr, nre, s