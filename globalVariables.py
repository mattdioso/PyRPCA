# File to hold ALL GLOBAL VARIABLES ONLY
import os
import math
import numpy as np

pathPrefix = "/Dev/SU-ECE-RESEARCH/PyRPCA_test/"

pathSuffix = "Madiyan_2012_Renamed"

INPUT_MAIN_FOLDER_NAME = os.path.join(pathPrefix, "DATA/Tajikistan_2012_CTPhotos_Renamed", pathSuffix)

OUTPUT_MAIN_FOLDER_NAME = os.path.join(pathPrefix, "RESULTS/Tajikistan_2012_CTPhotos_Renamed", pathSuffix)

MAX_SET_SIZE = 50

MIN_SET_SIZE = 2

SET_INFO_TEXT_FILE_NAME = "setInfo.txt"

# RPCA Parameters

lamda = 2 * math.exp(-2) # apparently 'lambda' is a keyword in python
epsilon = 5 * math.exp(-3) * np.linalg.norm(X, 'fro')
tau0 = 3 * math.exp(5)
SPGL1_tol = math.exp(-1)
tol = math.exp(-3)

optimizationFunctionMax = true
optimizationFunctionSum = not optimizationFunctionMax

RPCA_Parameters = {'lamda' : lamda, 'epsilon' : epsilon, 'tau0' : tau0, 'SPGL1_tol' : SPGL1_tol, 'tol' : tol, 'sum' : optimizationFunctionSum, 'max' : optimizationFunctionMax}

motionThreshold = 1
strelSize = 100
strelShape = 'disk'