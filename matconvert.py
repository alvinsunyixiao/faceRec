import cv2
import numpy as np
import os
import scipy.io as sio

dir = 'att_faces/'
dir2 = 'myFaces/'
#img = cv2.imread("/Users/alvinsun/Desktop/MIT-CBCL-facerec-database/test/0000_02287.pgm")

x1 = []
x2 = []
result = []
y = []

for s in os.listdir(dir2):
    if s=='.DS_Store':
        continue
    img = cv2.imread(dir2+s)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    vec = np.reshape(gray,(1,gray.size))[0]
    x1.append(vec)
    result.append(vec)
    y.append([1])

for sub in os.listdir(dir):

    for s in os.listdir(dir+sub):
        img = cv2.imread(dir+sub+'/'+s)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[15:102+5,:]
        img = cv2.resize(img,(90,90))
        vec = np.reshape(img,(1,img.size))[0]
        x2.append(vec)
        result.append(vec)
        y.append([2])

result = np.array(result)
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
fulldata = np.append(result,y,1)
myDict = {'X':result,'y':y,'fullData':fulldata,'x1':x1,'x2':x2}
sio.savemat('raw_data.mat',myDict)