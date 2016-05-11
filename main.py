import cv2
import numpy as np
import scipy.io as sio
from func import *

pca = sio.loadmat('pca_param.mat')
norm = sio.loadmat('norm_param.mat')
Theta = sio.loadmat('thetaBox.mat')
Theta1 = Theta['Theta1']
Theta2 = Theta['Theta2']
Theta3 = Theta['Theta3']
Theta1 = np.mat(Theta1)
Theta2 = np.mat(Theta2)
Theta3 = np.mat(Theta3)
mu = norm['mu']
sigma = norm['sigma']
U = pca['U']

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


faceCas = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cam = cv2.VideoCapture(0)
count = 1
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray,faceCas)
    if len(rects)==0:
        continue
    for x1,y1,x2,y2 in rects:
        faceim = im[y1:y2,x1:x2]
        resized = cv2.resize(faceim,(90,90))
        normx = convertImg(resized,mu,sigma)
        z = projectData(normx,U,400)
        hp = getProbability(z,Theta1,Theta2,Theta3)
        print hp

cam.release()