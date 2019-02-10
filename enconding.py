# coding:utf-8

# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--dataset", required=True,
	# help="path to input directory of faces + images")
ap.add_argument("-i", "--dataset", type = str,default ="./pos",
	help="path to input directory of faces + images")	
#ap.add_argument("-e", "--encodings", required=True,
#	help="path to serialized db of facial encodings")
ap.add_argument("-e", "--encodings", type=str,default="testhowardencodings.pickle",
 	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
# knownEncodings = np.array([])
# knownEncodings = np.zeros((2, 128))
knownEncodings = []
knownNames = np.array([])

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	# print("imagePath : ",imagePath)
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# print("rgb : ",rgb)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	# print("boxes : ",boxes )
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings

		knownEncodings.append(encoding)


# print(knownEncodings)






negap = argparse.ArgumentParser()
negap.add_argument("-i", "--dataset", type = str,default ="./neg",
	help="path to input directory of faces + images")
negap.add_argument("-e", "--encodings", type=str,default="testhowardencodings.pickle",
 	help="path to serialized db of facial encodings")
negap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(negap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
# negknownEncodings = np.zeros((7, 128))
negknownEncodings =[]
negknownNames = np.array([])

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	# print("imagePath : ",imagePath)
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# print("rgb : ",rgb)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	# print("boxes : ",boxes )
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings

	negknownEncodings.append(encoding)

# print("knownEncodings : ",knownEncodings)

print("[INFO] serializing encodings...")

label = np.array([[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0]])

# 3 合并数据
data = np.vstack((knownEncodings,negknownEncodings))
# print(data)
data = np.array(data,dtype='float32')
# print(data)

# 4 训练
# ml  机器学习模块 SVM_create() 创建
svm = cv2.ml.SVM_create() 

# 属性设置
svm.setType(cv2.ml.SVM_C_SVC) # svm type C_SVC
# svm.setType(cv2.ml.SVM_NU_SVC) # NU_SVC
# svm.setType(cv2.ml.SVM_ONE_CLASS ) # ONE_CLASS
# svm.setType(cv2.ml.SVM_EPS_SVR ) # EPS_SVR
# svm.setType(cv2.ml.SVM_NU_SVR ) # NU_SVR 

svm.setKernel(cv2.ml.SVM_LINEAR) # line
# svm.setKernel(cv2.ml.SVM_CUSTOM) # CUSTOM
# svm.setKernel(cv2.ml.SVM_POLY) # POLY
# svm.setKernel(cv2.ml.SVM_RBF) #RBF
# svm.setKernel(cv2.ml.SVM_SIGMOID) #SIGMOID
# svm.setKernel(cv2.ml.SVM_CHI2 ) #CHI2
# svm.setKernel(cv2.ml.SVM_INTER ) # INTER 
svm.setC(0.01)

# 训练
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)
print("Training OK!!")





pt_data = np.array([
	# [-0.00818702,  0.06719302,  0.04531664, -0.07009925, -0.13669556,
 #       -0.00798552, -0.06849691, -0.06066156,  0.14099972, -0.09220184,
 #        0.20620891, -0.08001547, -0.23477633,  0.01279183, -0.10045424,
 #        0.24627118, -0.22686793, -0.18489987, -0.0403777 ,  0.03025571,
 #        0.04008473,  0.11396039,  0.03479786,  0.02885284, -0.18262586,
 #       -0.39390516, -0.07373607, -0.05585312, -0.05219165, -0.06545386,
 #       -0.02192758,  0.10544245, -0.17343481,  0.04221106,  0.09633041,
 #        0.07055717,  0.04069887, -0.10633221,  0.18645541,  0.07948225,
 #       -0.32584959,  0.06932602,  0.10169229,  0.2746731 ,  0.13596734,
 #       -0.05565383,  0.0105849 , -0.13004786,  0.08523731, -0.16969651,
 #        0.02397175,  0.13899735,  0.03310945,  0.09334995, -0.01568829,
 #       -0.09557834,  0.02929383,  0.11363263, -0.11356723, -0.03319515,
 #        0.12558784, -0.05814943,  0.0625868 , -0.11964423,  0.17502588,
 #        0.0298284 , -0.09354193, -0.20438522,  0.06267527, -0.15090421,
 #       -0.114354  ,  0.03583977, -0.14919594, -0.13918027, -0.31096148,
 #       -0.04786228,  0.29958636,  0.12119626, -0.17706966,  0.08820503,
 #       -0.00624951,  0.01486148,  0.07894677,  0.16581149,  0.01256736,
 #        0.13483557, -0.08956102,  0.06382588,  0.28929955, -0.07977185,
 #       -0.00719017,  0.26011282, -0.01978891,  0.08726289,  0.05674425,
 #        0.01261386, -0.06672504,  0.03108343, -0.13361707,  0.04567478,
 #       -0.03905826,  0.00769592,  0.00485565,  0.1190499 , -0.1421015 ,
 #        0.11545012, -0.00774835,  0.00830678,  0.0035954 ,  0.09095208,
 #       -0.04471983, -0.10320811,  0.12212034, -0.19653746,  0.18965781,
 #        0.14874637,  0.0855585 ,  0.08730535,  0.10594481,  0.06590291,
 #       -0.07863092, -0.01565891, -0.18424456, -0.0081259 ,  0.08710009,
 #       -0.02914489,  0.09497213,  0.02810632]
	[-0.0792239 ,  0.02527511,  0.06113945, -0.10484602, -0.1000395 ,
       -0.06027834, -0.11885991, -0.10534794,  0.13432671, -0.17205729,
        0.20416042, -0.0881782 , -0.20410217, -0.01228404, -0.03666538,
        0.25368321, -0.19152215, -0.1869749 , -0.05096495,  0.03350271,
        0.02086278,  0.06204989,  0.00694811,  0.07670084, -0.14683954,
       -0.347332  , -0.13722052, -0.03128285, -0.07690658, -0.0315971 ,
       -0.03204339,  0.04106359, -0.13621172,  0.06747293,  0.06759074,
        0.08772819,  0.04139907, -0.06293508,  0.16065855,  0.02411251,
       -0.32830578,  0.09470147,  0.10473187,  0.21455559,  0.17570069,
       -0.00908622,  0.02374991, -0.16945277,  0.09306476, -0.15454701,
       -0.0110941 ,  0.13687903,  0.08746487,  0.05172459,  0.00663367,
       -0.09933111,  0.00510338,  0.15852441, -0.11145587, -0.01863184,
        0.08356327, -0.05500776, -0.00107406, -0.13799015,  0.20044389,
        0.07124326, -0.17667642, -0.22350448,  0.08395404, -0.18077436,
       -0.15203065,  0.04258388, -0.17077863, -0.17636424, -0.37014788,
       -0.05555705,  0.28092474,  0.0907533 , -0.21390361,  0.11847462,
        0.01649423,  0.03223427,  0.03363901,  0.19961169, -0.00661775,
        0.06729428, -0.05177454, -0.00383966,  0.26384833, -0.07566294,
        0.04370779,  0.19456172, -0.00556064,  0.1089546 , -0.0111677 ,
        0.03113803, -0.04636029, -0.00804299, -0.12507097,  0.0490169 ,
       -0.07641659,  0.03659654, -0.00597127,  0.08982665, -0.16869214,
        0.17652149,  0.00216362, -0.02088793, -0.02452124,  0.12768529,
       -0.0354049 , -0.07734733,  0.11257677, -0.23455496,  0.17341071,
        0.20742476,  0.09383309,  0.09099693,  0.11012925,  0.09989384,
       -0.07712739,  0.05590513, -0.19390234, -0.02288963,  0.07452865,
       -0.07497701,  0.09558274,  0.03877435]
       ])
pt_data = np.array(pt_data,dtype='float32')
(par1,par2) = svm.predict(pt_data)
print("the result of prediction  : " + str(int(par2[0])))

