import os
import numpy as np 
from globalVariables import *
import csv
from getFolders import getFolders
from createSets import createSets
from get_setInfo import get_setInfo
from getImageData import getImageData
from run_RPCA import run_RPCA
from threshold_RPCA import threshold_RPCA
from morph_RPCA import morph_RPCA

os.mkdir(OUTPUT_MAIN_FOLDER_NAME)

# MATLAB: load()
if (os.path.isfile(os.path.join(OUTPUT_MAIN_FOLDER_NAME, 'folderList.csv'))):
	with open(os.path.join(OUTPUT_SHARE_FOLDER_NAME, 'folderList.csv'), 'rb') as f:
		reader = csv.reader(f)
		folderList = list(reader)
else:
	print('folderList DNE. Getting Folders now \n')
	folderList, numFolders, emptyFolderList = getFolders()
	print('DONE\n')

# MATLAB: save()
folderListFile = OUTPUT_MAIN_FOLDER_NAME + 'folderList.csv'
emptyFolderListFile = OUTPUT_MAIN_FOLDER_NAME + 'emptyFolderList.csv'
with open(folderListFile, "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for val in folderList:
		writer.writerow([val])

with open(emptyFolderListFile, "w") as out:
	writer = csv.writer(out, lineterminator='\n')
	for val in emptyFolderList:
		writer.writerow([val])

#Create temporal sets
for i in range(0, numFolders):
	folderName = folderList(i).name 

	if not os.listdir(os.path.join(OUTPUT_MAIN_FOLDER_NAME, folderName, SET_INFO_TEXT_FILE_NAME)):
		print("\nCreating Sets\nFolder: %s\n(%d out of %d)\n", folderName, f, numFolders)
		createSets(folderName)
		setInfo = get_setInfo(os.path.join(OUTPUT_MAIN_FOLDER_NAME, folderName))
		csvfile = OUTPUT_MAIN_FOLDER_NAME + '/' + folderName + '/setInfo.csv'
		with open(csvfile, "w") as output:
			writer = csv.writer(output, lineterminator='\n')
			for val in setInfo:
				wrtier.writerow([val])

for i in range(0, numFolders):
	folderName = foderList(i).name
	directoryOfSets = os.listdir(os.path.join(OUTPUT_MAIN_FOLDER_NAME, folderName, 'Set_*'))
	for x in range(0, len(directoryOfSets)):
		thisSet = directoryOfSets[x]
		imageDataLocation = os.path.join(OUTPUT_SHARE_FOLDER_NAME, folderName, thisSet.name)
		X, imageHeight, imageWidth = getImageData(imageDataLocation)

		#Run RPCA
		rpca_results = run_RPCA(X, imageHeight, imageWidth, RPCA_Parameters)

		#Motion Thresholding
		rpca_results = threshold_RPCA(rpca_results, motionThreshold)

		#Binary morphology
		rpca_results = morph_RPCA(rpca_results, strelSize, strelShape)

		#Save RPCA results
		rpca_results_fileName = os.path.join(OUTPUT_SHARE_FOLDER_NAME, folders, thisSet.name, 'rpca_results.csv')

		with open(rpca_results_fileName, "w") as output:
			writer = csv.writer(output, lineterminator='\n')
			for val in rpca_results:
				writer.writerow([val])