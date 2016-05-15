import cv2
import numpy as np
import os,time
from person import Person
import picamera
cam = picamera.PiCamera()
cam.resolution = (640,480)

from person import recgonize

try:
    while 1:
        result = recgonize(cam)
        if result == None:
            continue
        (name, confidence) = result
        print name + '    ' + str(confidence)
except KeyboardInterrupt:
    cam.close()
