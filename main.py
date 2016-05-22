import cv2
import numpy as np
import os,time
from person import Person
import serial
import picamera
cam = picamera.PiCamera()
cam.resolution = (640,480)
ser = serial.Serial('/dev/ttyACM0',9600)
from person import recgonize
thresh = 65
try:
    while 1:
        result = recgonize(cam)
        if result == None:
            print 'no face'
            ser.write('noface\n')
            continue
        (name, confidence) = result
        if confidence<=thresh:
            ser.write('unkown\n')
        else:
            ser.write(name+'\n')
        print name + '    ' + str(confidence)
except KeyboardInterrupt:
    cam.close()
    ser.close()
