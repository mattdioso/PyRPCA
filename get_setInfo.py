import os
import numpy as np 
import csv

def get_setInfo(filePath, defName="setInfo.txt"):
	if len(locals()) < 2:
		txtFileDir = os.listdir(filePath)

		txtList = []
		for item in txtFileDir:
			if item.endswith(".txt"):
				txtList.append(item)

		if len(txtList) > 1:
			print("Multiple .txt files exist in %s. Opening setInfo.txt", filePath)
			cpfn = filePath + '\setInfo.txt'
		else:
			cpfn = filePath + txtList(1).name 
	else:
		cpfn = filePath + '\\' + defName

	with open(cpfn, 'r') as input:
		setInfo = {'set' : 0, 'nImgs' : 0, 'names' : cell(1), 'note' : []}
		nSets = input.readline()
		setN = 0
		imgNum = 0
		for line in input:
			if line.endswith("JPG"):
				imgNum = imgNum = imgNUm + 1
				setInfo[setN].names[imgNum] = line
			else:
				setN = setN + 1
				info = line.split(" ")
				s = info[0]
				n = info[1]
				setInfo[setN].set = int(s)
				setInfo[setN].nIMgs = int(n)
				imgNum=0

	return setInfo