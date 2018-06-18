import os
import scipy
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from globalVariables import *

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def getImageData(imageFolder, *args):
	if len(locals()) < 2:
		ext='JPG'
	else:
		if args[1].lower() == 'extension':
			ext=args[2]
		else:
			ext = args[1]

	imageDirectory = os.listdir(imageFolder)

	for x in range(1, len(imageDirectory)):
		temp = rgb2gray(mpimg.imread(imageFolder + '\\' + imageDirectory(x).name))
		temp.asType(np.int8)
		X(:,q) = temp(:) #LOOK INTO THIS

	r, c = temp.shape
	return X, r, c