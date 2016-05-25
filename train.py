import cv2
import numpy as np
import os,time
from person import Person
#import picamera
import sys
cam = cv2.VideoCapture(0)
'''
cam.resolution = (640,480)
cam = cv2.VideoCapture(0)
'''
train_name = 'Su'
aperson = Person(train_name,cam)
aperson.train()
