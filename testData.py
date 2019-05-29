# coding:utf-8
from imutils import paths
from picamera import PiCamera, Color
import face_recognition
import cv2
import os
import numpy as np
import sys
import time
import csv
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
camera.resolution =(240,240)
camera.brightness = 60
camera.framerate = 1

#previously import user RFID
RFIDData = []



def RFIDInput():
    with open('information.csv',newline = '') as csvFile:
        rows = csv.reader(csvFile)
        for row in rows:
            RFIDData.append(row)
        


def test():
    camera.start_preview()
    camera.preview.fullscreen = False
    camera.preview.window = (750,500,600,600)
    while(True):
        
        #use RFID to enter sign in
        #Request tag
        (error,dataasd) = rdr.request()
        #if not error:        
        print("You can use your card")
        (error,getuid) = rdr.anticoll()    
        
        if not error:
            totalop = time.time()
            
            
            useRFIDop = time.time()
            print ("Card read UID: "+str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3]))
            uid = str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3])
            useRFIDed = time.time()
            
            
            #take picture
            print("taking picture")
            
            #preview  the camera for user to see himself
            cameraUseop = time.time()
            
            camera.capture('./'+uid+'.jpg')
            cameraUseed = time.time()
            cameraUse = cameraUseed - cameraUseop
            
            
            
            #encoding picture
            readImageop = time.time()
            image = cv2.imread('./'+uid+'.jpg')
            readImageed = time.time()
            readImage = readImageed - readImageop
            
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
            encodingop = time.time()
            encodings = face_recognition.face_encodings(rgb, boxes)
            encodinged = time.time()
            encoding = encodinged- encodingop
            
            
            #print(encodings[0][0])
            
            
            
            #if camera can't encode any face
            entersvmop = time.time()
            if encodings:
                #input predict model and output the result
                pt_data = np.array(encodings,dtype='float32')
                for rfidData in RFIDData:
                    if rfidData[1] == uid:
                        if rfidData[0] == "howard":
                            (par1,par2) = svmhoward.predict(pt_data)
                            isTrue(par2[0][0])
                        elif rfidData[0] == "laio":
                            (par1,par2) = svmlaio.predict(pt_data)
                            isTrue(par2[0][0])
                        elif rfidData[0] == "oscar":
                            (par1,par2) = svmoscar.predict(pt_data)
                            isTrue(par2[0][0])
                        elif rfidData[0] == "tingju":
                            (par1,par2) = svmtingju.predict(pt_data)
                            isTrue(par2[0][0])
                        elif rfidData[0] == "xin":
                            (par1,par2) = svmxin.predict(pt_data)
                            isTrue(par2[0][0])
                        elif rfidData[0] == "yancheng":
                            (par1,par2) = svmyancheng.predict(pt_data)
                            isTrue(par2[0][0])
                        elif rfidData[0] == "yuchen":
                            (par1,par2) = svmyuchen.predict(pt_data)
                            isTrue(par2[0][0])
                
                ''''
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
                '''
            else :
                print()
                print("please try again")
                print()
            entersvmed = time.time()
            entersvm = entersvmed - entersvmop
            totaled = time.time()
            total = totaled - totalop
            
            print()
            print("cameraUse, readImage, location, encoding,  entersvm,  total time")
            print(cameraUse,readImage,location,encoding,entersvm,total )
            print("cameraUse : ",cameraUse)
            print("readImage : ",readImage)
            print("location : ",location)
            print("encoding : ",encoding)
            print("entersvm : ",entersvm)
            print("total time : ", total)
            print()
        time.sleep(1)
    camera.stop_preview()        
            
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
    RFIDInput()
    test()