import cv2
import numpy as np
import os,time
from person import Person

aperson = Person('Nancy Wang')
aperson.train()

'''
from person import recgonize
cam = cv2.VideoCapture(0)
for i in range(10):
    result = recgonize(cam)
    if result==None:
        continue
    (name, confidence) = result
    print name + '    ' + str(confidence)
'''