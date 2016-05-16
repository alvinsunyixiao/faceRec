import cv2
import numpy as np
import os,time
from person import Person
import picamera
import sys
cam = picamera.PiCamera()
cam.resolution = (640,480)
train_name = sys.argv[1]
aperson = Person(train_name,cam)
aperson.train()
