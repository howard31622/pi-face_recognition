#!/usr/bin/python
from picamera import PiCamera, Color
import time
import os
import cv2
import sys
import getpass
import csv 
sys.path.insert(0, "/home/pi/pi-rc522/ChipReader")
from pirc522 import RFID



#個人資料 username 為學號 ,password為密碼,uid為RFID
username = ""
password = ""
uid = ""

video_name = "GN_Max"
video_format = 'h264'
video_row_size = 320
video_col_size = 320
#camera default setting
camera = PiCamera()
#camera.resolution = (r_size, c_size)
camera.resolution =(320,320)
camera.brightness = 60
camera.framerate = 1
camera.preview.fullscreen = False
camera.preview.window = (750,10,320,320)
#camera.preview.window = (750, 10, r_size, c_size)
camera.annotate_text_size = 120
camera.annotate_foreground = Color('black')
#RFID default setting
rdr = RFID()
util = rdr.util()
util.debug = False




def roll_call(v_name,v_format):
    
    while(True):
        test = "1"
        uid = "1"
        input("please push enter ")
        while test == "1":
            #Request tag
            (error,dataasd) = rdr.request()

             #if not error:
            
            print("You can use your card")
            (error,getuid) = rdr.anticoll()    
            
            if not error:
                
                
                #camera.annotate_text = "3"
                #print('3')
                #time.sleep(1)
                #camera.annotate_text = "2"
                #print('2')
                time.sleep(1)
                #camera.annotate_text = "1"
                #print('1')
                #time.sleep(0.5)
                #camera.annotate_text = ""
                #print("start_recording")
                
                camera.start_preview()

                print ("Card read UID: "+str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3]))
                uid = str(getuid[0])+","+str(getuid[1])+","+str(getuid[2])+","+str(getuid[3])
                video_name = out_time + "_" + uid
                camera.start_recording(video_name + '.' + v_format, quality=23)
                print("please enter your username and password")
                username = input('username :')
                #getpass.getpass
                password = getpass.getpass('password :')
                write_information()
                #input()
                
                #camera.wait_recording(15.04)
                camera.stop_recording()
                camera.stop_preview()
                #print("stop_recording")
                video2image(video_name, video_format)
                print("Thank you")
                
                test = "2"
                
                #print("Start to take picture")

                #for i in range(0,50):
                 #   camera.capture('%s.jpg'%i)
                #vc =cv2.VideoCapture('video.h264')                
                #c=1
                #if vc.isOpened():
                #    rval,frame=vc.read()
                #else:
                #    rval = False
#尚未完成
def write_information():
    print(uid, username, password)
    with open('./information.csv','w',newline = '') as csv_file :
        csv_writer = csv.writer(csv_file)
        
        csv_writer.writerow([uid,username,password])
    




def video2image(v_name, v_format):
    img_out_dir = "./output_" + v_name
    if not os.path.isdir(img_out_dir):
        os.mkdir(img_out_dir)

    video_cap = cv2.VideoCapture(v_name + '.' + v_format)
    success, image = video_cap.read()
    if not success:
        print("read video error.")
    else:
        print("read video success.")
    print("start doing video to image.")
    frame_count = 0
    while success:
        cv2.imwrite(img_out_dir + "/" + v_name + "_frame_%d.png" % frame_count, image)  # save frame as JPEG file
        success, image = video_cap.read()
        frame_count += 1
    print("total frame number: %d." % frame_count)


if __name__ == "__main__":
   
    #print("User name:", video_name)
    out_time = "D" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    video_name = out_time + "_" + video_name
    roll_call(video_name,video_format)
    #cap_video(video_name, video_format, video_row_size, video_col_size)
    #video2image(video_name, video_format)
 