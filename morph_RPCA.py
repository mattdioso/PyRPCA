import numpy as np 
import os
import scipy
import cv2
from rpca_reshape import rpca_reshape

def morph_RPCA(res, strelSize=100, strelShape='disk'):
	if strelSize % 1 > 0:
		print ("[morph_RPCA] %f is an invalid structural element size. Use only positive integers", strelSize)
		return

	N = 1

	se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, strelSize)

	res.M = np.zeros(res[0] * res[1], res.setSize)

	for i in range(0, res.setSize):
		templace = rpca_reshape(res, 'T', i)
		image = rpca_reshape(res, 'X', i)

		kernel = np.ones((5, 5), np.uint8)
		m1 = cv.erode(template, kernel, i)

		m2 = cv.morphologyEx(m1, cv.MORPH_CLOSE, se)

		res.M[:, i] = m2[:]

	res.M = bool(res.M) #CHECK THIS

	return res