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
knownEncodings = np.array([])
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
		knownEncodings = np.append(knownEncodings,encoding)
		# knownEncodings.append(encoding)
		knownNames = np.append(knownNames,name)
		# knownNames.append(name)
		# print(encoding)

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
negknownEncodings = np.array([])
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
		negknownEncodings = np.append(negknownEncodings,encoding)
		# negknownEncodings.append(encoding)
		negknownNames = np.append(negknownNames,name)
		# negknownNames.append(name)
		# print(encoding)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
# data = {"encodings": knownEncodings, "names": knownNames}
# print(data)
# negdata = {"encodings": negknownEncodings, "names": negknownNames}
# print(negdata)
#print (" data : ",data)
# f = open(args["encodings"], "wb")
# f.write(pickle.dumps(data))
# f.close()
print(knownEncodings)
data = np.vstack((knownEncodings,negknownEncodings))
print(data)
label = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

# 3 合并数据
# data = np.vstack((rand1,rand2))
# data = np.array(data,dtype='float32')
# print(data)
# 4 训练
# ml  机器学习模块 SVM_create() 创建
svm = cv2.ml.SVM_create() 

# 属性设置
svm.setType(cv2.ml.SVM_C_SVC) # svm type
svm.setKernel(cv2.ml.SVM_LINEAR) # line
svm.setC(0.01)

# 训练
result = svm.train(data,cv2.ml.ROW_SAMPLE,label)

