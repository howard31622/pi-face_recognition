#!/usr/bin/python
import  picamera
import  time

for take in range(0,10):
	camera = picamera.PiCamera()
	camera.resolution =  (640,480) 
	time.sleep(2)
	camera.capture('%s.jpg'%(take))
