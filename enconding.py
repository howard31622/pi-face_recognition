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
# import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--dataset", type = str,default ="./t",
	# "./trainingpicture/database/TMP/training_data/testdata/yuchen",
	help="path to input directory of faces + images")	
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the list of known encodings and known names
knownEncodings = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	# print("imagePath : ",imagePath)
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# print("rgb : ",rgb)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
	# print("boxes : ",boxes )
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		# print(encoding)
		knownEncodings.append(encoding)


# labelpos = int(len(imagePaths))
# print(labelpos)
# print(knownEncodings)





#./trainingpicture/database/TMP/training_data/False_data
negap = argparse.ArgumentParser()
negap.add_argument("-i", "--dataset", type = str,default ="./u",
	help="path to input directory of faces + images")
negap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(negap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
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
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		# print(encoding)
		negknownEncodings.append(encoding)
labelneg = int(len(imagePaths))
# print("knownEncodings : ",knownEncodings)

print("[INFO] serializing encodings...")

# labela = np.array([[]])
# labela = np.ones((labelpos,1))
# for i in range(0,labelpos):
# 	labela[i] = 1
# print(labela)

# labelb = np.array([[]])
# labelb = np.zeros((labelneg,1))
# for i in range(0,labelneg):
# 	labelb[i] = 0 
# print(labelb)
# label = np.vstack((labela,labelb))
# print(label)
# label = np.array([
# 	[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
# 	[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]
# 	])
label = np.array([
	[1],[0]
	])
# label = np.array(
# 	[[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
# 	[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
# 	)
# print(label)
# 3 合并数据
data = np.vstack((knownEncodings,negknownEncodings))
print(data)
data = np.array(data,dtype='float32')
# print(data)

# 4 遜賃
# ml  機器學習model SVM_create() 創建
svm = cv2.ml.SVM_create() 

# 屬性設置
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

# 訓練
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)
print("Training OK!!")





#要預測的資料
testgap = argparse.ArgumentParser()
testgap.add_argument("-i", "--dataset", type = str,default ="./v",
	help="path to input directory of faces + images")
testgap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
testargs = vars(testgap.parse_args())

testknownEncodings =[]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
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
		# print(encoding)
		testknownEncodings.append(encoding)
pt_data = np.vstack((testknownEncodings))		




       # ])
pt_data = np.array(pt_data,dtype='float32')
# print(pt_data)
(par1,par2) = svm.predict(pt_data)
# print((par1,par2))
# print("the result of prediction  : " + str(int(par2[0])))
print("the result of prediction  : " )
print(par2)
