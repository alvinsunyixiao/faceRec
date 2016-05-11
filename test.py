import cv2
import numpy as np
import os,time

dir = 'att_faces/'
dir2 = 'myFaces/'
#img = cv2.imread("/Users/alvinsun/Desktop/MIT-CBCL-facerec-database/test/0000_02287.pgm")

for s in os.listdir(dir2):
    img = cv2.imread(dir2+s)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    vec = np.reshape(gray,(1,gray.size))[0]
    print vec


'''
for sub in os.listdir(dir):

    for s in os.listdir(dir+sub):
        img = cv2.imread(dir+sub+'/'+s)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[15:102+5,:]
        img = cv2.resize(img,(90,90))
        cv2.imshow('go',img)
        key = cv2.waitKey(0)
        if key==27:
            break
    if key==27:
        break
cv2.destroyAllWindows()
'''