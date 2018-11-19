#!/usr/bin/python
import  picamera
import  time

camera = picamera.PiCamera()
camera.resolution =  (640,480) 
print("Start to take picture")
time.sleep(1)
for i in range(20,30):
	print("picture %d"%(i))
	#time.sleep(2)
	#camera.capture('howard.png')
	camera.capture('howard%s.png'%(i))

