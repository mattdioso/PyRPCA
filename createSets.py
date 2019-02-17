from globalVariables import *
import os
import time
import numpy as np

def createSets(folder):

	inputFolderName = os.path.join(INPUT_MAIN_FOLDER_NAME, folder, "\\")

	outputFolderName = os.path.join(OUTPUT_FOLDER_NAME, folder, "\\")

	setInfoFileName = os.path.join(OUTPUT_FOLDER_NAME, 'setInfo.txt')

	timeLapseThreshold = 90

	fileList = os.listdir(os.path.join(INPUT_MAIN_FOLDER_NAME, "*.JPG"))
	numIm = len(fileList)

	rejectedIm = []
	for i in range(0, numIm):
		if fileList[i].bytes > 0:
			try:
				timeVecCurrent[i] = time.ctime(os.path.getmtime(os.path.join(INPUT_MAIN_FOLDER_NAME, fileList(i).name)))
			except OSError as e:
				timeVecCurrent[i] = float('nan')
				rejectedIm = os.path.join(rejectedIm, i)
				print("rejected image: %s (i=%d)\n", fileList[i], i)
		else:
			rejectedIm = [rejectedIm i]
			timeVecCurrent[i] = float('nan')
			print("rejected image: %s (i=%d)\n", fileList[i], i)

	fileList[rejectedIm]= []
	timeVecCurrent[rejectedIm] = []
	sortedTime, index = np.sort(timeVecCurrent)
	fileList = fileList[index]
	numIm = len(fileList)

	timeVecPrevious = numpy.zeros(6)
	setInfo = numpy.zeros(numIm)
	k = 0

	for i in range(0, numIm):
		if setInfo[i] == 0

		imInfo = 