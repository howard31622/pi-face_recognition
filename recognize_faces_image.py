# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2
import time
import picamera
import sys
sys.path.insert(0, "/home/pi/pi-rc522/ChipReader")
from pirc522 import RFID
import signal


finish = 1


preStart = time.time()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", type=str, default="jackyencodings.pickle" ,
	help="path to serialized db of facial encodings")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
ap.add_argument("-i", "--image", type=str, default="examples/jackytest.png",
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
preEnd = time.time()
pre = preEnd - preStart
print("pre : ", pre)


# load the known faces and embeddings
loadFaceTimeStart = time.time()
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
loadFaceTimeEnd = time.time()
loadFaaceTime = loadFaceTimeEnd - loadFaceTimeStart
print("load Faace and Embeddings  : ",loadFaaceTime)


# load the input image and convert it from BGR to RGB
loadInputImageStart = time.time()
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
loadInputImageEnd = time.time()
loadInputImage = loadInputImageEnd - loadInputImageStart
print("load Input Image and Convert it from BGR to RGB : ",loadInputImage)

#while finish == 1:
	rdr = RFID()
	util = rdr.util()
	util.debug = False
while True:
    	#Request tag
    # (error, data) = rdr.request()
    		if not error:
         	print ("\nDetected")
         (error, uid) = rdr.anticoll()
    #     if not error:
            #Print UID
            # print ("Card read UID: "+str(uid[0])+","+str(uid[1])+","+str(uid[2])+","+str(uid[3]))
            # time.sleep(1)



	#take picture 
	# camera = picamera.PiCamera()
	# camera .resolution = (640,480)
	#time.sleep(2)
	#camera.capture()


	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
	recognizingStart = time.time()
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	recognizingEnd = time.time()
	recognizing = recognizingEnd - recognizingStart 
	print("recognizing faces : " ,recognizing)
	# initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	facialEmbeddingsStart = time.time()
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding ,tolerance=0.8)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
	
			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)
		print("name : " ,name)
	facialEmbeddingsEnd = time.time()
	facialEmbeddings = facialEmbeddingsEnd -facialEmbeddingsStart
	print("loop over the facial embeddings : ", facialEmbeddings)







	# loop over the recognized faces
	faceRecognizedStart = time.time()
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
	faceRecognizedEnd = time.time()
	faceRecognized = faceRecognizedEnd  - faceRecognizedStart 
	print("loop over the recognized faces : ",faceRecognized)





	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(5000)
	allFinish = time.time()
	final = allFinish - preStart
	print("running the program time is ", final)
 
