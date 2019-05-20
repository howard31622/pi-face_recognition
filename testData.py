# coding:utf-8
from imutils import paths
from picamera import PiCamera, Color
import face_recognition
import cv2
import os
import numpy as np
import sys
import time
sys.path.insert(0, "/home/pi/pi-rc522/ChipReader")
from pirc522 import RFID

# loading model
svm = cv2.ml.SVM_load("howard_svm_model.mat")

#RFID default setting
rdr = RFID()
util = rdr.util()
util.debug = False

#PiCamera default setting
camera = PiCamera()
camera.resolution =(320,320)
camera.brightness = 60
camera.framerate = 1






#x =0
def test():
    while(True):
        #use RFID to enter sign in
        #Request tag
        (error,dataasd) = rdr.request()
        #if not error:        
        print("You can use your card")
        (error,getuid) = rdr.anticoll()    
        #if x == 0 :
        if not error:    
            print ("Card read UID: "+str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3]))
            uid = str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3])
#            uid = "zzzz"
            #take picture
            camera.capture('./'+uid+'.jpg')
            #encoding picture
            image = cv2.imread('./'+uid+'.jpg')
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
            # print("rgb : ",rgb)
            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb,model="hog")
            # print("boxes : ",boxes )
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            #print(encodings)
            #input predict model and output the result
            pt_data = np.array(encodings,dtype='float32')
            (par1,par2) = svm.predict(pt_data)
            #print("the result of prediction  : " )
            if str(int(par2[0][0])) == "1" :
                print("")    
                print("The user is correct")
                print("")
            else :
            
                print("The user is error ")
            time.sleep(5)
            #print(par2[0][0])
    
    #print("testing")
    
if __name__ == "__main__":
    test()