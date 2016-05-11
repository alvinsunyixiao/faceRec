import cv2
import numpy as np
import scipy.io as sio

count = 136
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

faceCas = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
cam = cv2.VideoCapture(0)

while True:
    ret,im = cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    rects = detect(gray,faceCas)
    if (len(rects)==0):
        continue
    for x1,y1,x2,y2 in rects:
        faceim = im[y1:y2,x1:x2]
        resized = cv2.resize(faceim,(90,90))
        cv2.imwrite('myFaces/img%d.jpg'%count,resized)
        count += 1
        cv2.imshow('go',resized)
        key = cv2.waitKey(1)
        if key==27:
            break
    if key==27:
        break
