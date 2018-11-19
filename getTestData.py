#!/usr/bin/python
import  picamera
import  time

camera = picamera.PiCamera()
camera.resolution =  (640,480) 
print("Start to take test picture")

#get test data picture 
camera.capture('jackytest.png')

