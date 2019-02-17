from globalVariables import *
import os
import numpy as np

def getFolders: 
	folderList = os.listdir(INPUT_MAIN_FOLDER_NAME)
	numEmptyFolders=0
	emptyFolders = {}

	while folderList[0].name[0] == '.':
		folderList[0] = []

	numFolders = len(folderList)

	i=1

	while i <= numFolders
		if os.path.isdir(folderList[i]):
			print("Prcessing folder %i out of %i (%s)\n", i, numFolders, folderList[i].name)
			folderMain = folderList[i].name
			listing = os.listdir(os.path.join(INPUT_MAIN_FOLDER_NAME, folderMain, "\\"))

			try:
				while listing[1].name[1] == '.':
					listing[1] = []
			except OSError as e:
				print("Folder number %s (%s) has no files. Skipping", str(i), folderList[i].name)
				numEmptyFolders = numEmptyFolders + 1
				emptyFolders[numEmptyFolders] = folderList[i].name
				i = i +1
				continue

			fileNames = os.listdir(os.path.join(INPUT_MAIN_FOLDER_NAME, folderMain, '\*.JPG'))

			if (len(listing) != len(fileNames)):
				for j in range(0, len(listing)):
					if (os.path.isdir(list(j))):
						listing[j].name = os.path.join(folderMain, '\\', listing[j].name)
						folderList(len(folderList)+1) = listing[j]
						print("Adding folder %s\n", listing[j].name)
						numFolders = numFolders + 1
		i = i + 1

	imagesExist = []
	for i in range(0, numFolders):
		folderMain = folderList[i].name
		if len(os.path.isdir(os.path.join(INPUT_MAIN_FOLDER_NAME, folderMain, '\*.JPG'))):
			imagesExist[i] = 1

	i = np.where(imagesExist==0)
	folderList[i] = []
	numFolders = len(folderList)

	return folderList, numFolders, emptyFolders