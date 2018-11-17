#!/usr/bin/python
import  picamera
import  time

camera = picamera.PiCamera()
camera.resolution =  (640,480) 
for i in range(0,10):

	time.sleep(2)
	#camera.capture('howard.jpg')
	camera.capture('howard%s.jpg'%(i))
