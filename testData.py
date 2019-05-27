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
#svmginger = cv2.ml.SVM_load("ginger_svm_model.mat")
svmhoward = cv2.ml.SVM_load("howard_svm_model.mat")
svmlaio = cv2.ml.SVM_load("laio_svm_model.mat")
svmoscar = cv2.ml.SVM_load("oscar_svm_model.mat")
svmtingju = cv2.ml.SVM_load("tingju_svm_model.mat")
svmxin = cv2.ml.SVM_load("xin_svm_model.mat")
svmyancheng = cv2.ml.SVM_load("yancheng_svm_model.mat")
svmyuchen = cv2.ml.SVM_load("yuchen_svm_model.mat")

#RFID default setting
rdr = RFID()
util = rdr.util()
util.debug = False

#PiCamera default setting
camera = PiCamera()
camera.resolution =(320,320)
camera.brightness = 60
camera.framerate = 1







def test():
    while(True):
        
        #use RFID to enter sign in
        #Request tag
        (error,dataasd) = rdr.request()
        #if not error:        
        print("You can use your card")
        (error,getuid) = rdr.anticoll()    
        
        if not error:
            
            useRFIDop = time.time()
            print ("Card read UID: "+str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3]))
            uid = str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3])
            useRFIDed = time.time()
            
            
            #take picture
            print("taking picture")
            
            #preview  the camera for user to see himself
            cameraUseop = time.time()
            camera.start_preview()
            camera.preview.fullscreen = True
            camera.capture('./'+uid+'.jpg')
            cameraUseed = time.time()
            cameraUse = cameraUseed - cameraUseop
            print("cameraUse : ",cameraUse)
            
            
            #encoding picture
            encodingaop = time.time()
            image = cv2.imread('./'+uid+'.jpg')
            encodingaed = time.time()
            encodinga = encodingaed - encodingaop
            print("encodinga : ",encodinga)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # print("rgb : ",rgb)
            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            locationop = time.time()
            boxes = face_recognition.face_locations(rgb,model="hog")
            locationed = time.time()
            location = locationed - locationop
            #print("boxes : ",boxes )
            
            
            
            # compute the facial embedding for the face
            encodingbop = time.time()
            encodings = face_recognition.face_encodings(rgb, boxes)
            encodingbed = time.time()
            
            #print(encodings[0][0])
            
            camera.stop_preview()
            
            #if camera can't encode any face
            if encodings:
                #input predict model and output the result
                pt_data = np.array(encodings,dtype='float32')
                inputSvm = input("Please intput your name. For example : ginger ,howard ,laio ,oscar ,tingju ,xin ,yancheng ,yuchen : ")
                if inputSvm == "howard":
                    (par1,par2) = svmhoward.predict(pt_data)
                    isTrue(par2[0][0])
                elif inputSvm == "laio":
                    (par1,par2) = svmlaio.predict(pt_data)
                    isTrue(par2[0][0])
                elif inputSvm == "oscar":
                    (par1,par2) = svmoscar.predict(pt_data)
                    isTrue(par2[0][0])
                elif inputSvm == "tingju":
                    (par1,par2) = svmtingju.predict(pt_data)
                    isTrue(par2[0][0])
                elif inputSvm == "xin":
                    (par1,par2) = svmxin.predict(pt_data)
                    isTrue(par2[0][0])
                elif inputSvm == "yancheng":
                    (par1,par2) = svmyancheng.predict(pt_data)
                    isTrue(par2[0][0])
                elif inputSvm == "yuchen":
                    (par1,par2) = svmyuchen.predict(pt_data)
                    isTrue(par2[0][0])
                #print("the result of prediction  : " )
            
            else : 
                print("please try again")
                
            
            
            
            #print(par2[0][0])
    
    #print("testing")

#to test the predict model is truev

def isTrue(result):
    if str(int(result)) == "1" :
        print("")    
        print("The user is correct")
        print("")
    else :
            
        print("The user is error ")
        time.sleep(3)
    
    
if __name__ == "__main__":
    test()