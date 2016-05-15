import cv2
import numpy as np
import os,time
from person import Person
import picamera
cam = picamera.PiCamera()
cam.resolution = (640,480)
'''
aperson = Person('Alvin',cam)
aperson.train()

'''
from person import recgonize

try:
    for i in range(10):
        result = recgonize(cam)
        if result==None:
            continue
        (name, confidence) = result
        print name + '    ' + str(confidence)
except KeyboardInterrupt:
    cam.close()
