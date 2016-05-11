import cv2


cam = cv2.VideoCapture(0)

count = 1

while True:
    ret, img = cam.read()
    cv2.imshow('go',img)
    key = cv2.waitKey(1)
    if key==99:
        cv2.imwrite('momFace/img%d.jpg'%count,img)
        count += 1
    elif key==27:
        break