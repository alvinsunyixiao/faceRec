# -*- coding: utf-8 -*-

API_KEY = 'ae397874d1b74db82804dcfb67d7a249'
API_SECRET = 'kDH7U5ggdkDLPSLy27Tz-eXRZWJwTOKm'

# Import system libraries and define helper functions
# 导入系统库并定义辅助函数
import time
from pprint import pformat
import cv2
import numpy as np

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

def print_result(hint, result):
    def encode(obj):
        if type(obj) is unicode:
            return obj.encode('utf-8')
        if type(obj) is dict:
            return {encode(k): encode(v) for (k, v) in obj.iteritems()}
        if type(obj) is list:
            return [encode(i) for i in obj]
        return obj
    print hint
    result = encode(result)
    print '\n'.join(['  ' + i for i in pformat(result, width = 75).split('\n')])

# First import the API class from the SDK
# 首先，导入SDK中的API类
from facepp import API,File

api = API(API_KEY, API_SECRET)

while True:
    ret, img = cam.read()
    img = cv2.resize(img,(640,360))
    cv2.imwrite('buf.jpg',img)
    result = api.recognition.identify(img = File('buf.jpg'), group_name = 'test')
    #print_result('',result)
    if len(result['face'])==0:
        print 'no face\n'
        continue
    print 'The person with highest confidence:', \
        result['face'][0]['candidate'][0]['person_name']
    myface = result['face'][0]
    name = myface['candidate'][0]['person_name']
    conf = myface['candidate'][0]['confidence']
    face_id = myface['face_id']
    #api.person.add_face(person_name = name, face_id = face_id)
    rects = []
    center = (myface['position']['center']['x'],myface['position']['center']['y'])
    height = myface['position']['height']
    width = myface['position']['width']
    x1 = int((center[0]-width/2)*img.shape[1]/100)
    x2 = int((center[0]+width/2)*img.shape[1]/100)
    y1 = int((center[1]-height/2)*img.shape[0]/100)
    y2 = int((center[1]+height/2)*img.shape[0]/100)
    rects.append((x1,y1,x2,y2))
    draw_rects(img,rects,(0,255,0))
    rects = []
    center = (myface['position']['eye_left']['x'],myface['position']['eye_left']['y'])
    x1 = int(center[0]*img.shape[1]/100)-10
    x2 = x1+20
    y1 = int(center[1]*img.shape[0]/100)-10
    y2 = y1+20
    rects.append((x1,y1,x2,y2))
    center = (myface['position']['eye_right']['x'],myface['position']['eye_right']['y'])
    x1 = int(center[0]*img.shape[1]/100)-10
    x2 = x1+20
    y1 = int(center[1]*img.shape[0]/100)-10
    y2 = y1+20
    rects.append((x1,y1,x2,y2))
    draw_rects(img,rects,(255,0,0))
    cv2.putText(img,name+'   '+str(conf),(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0))
    cv2.imshow('go',img)
    key = cv2.waitKey(1)
    if key==27:
        break

result = api.recognition.train(group_name = 'test', type = 'all')
session_id = result['session_id']
while True:
    result = api.info.get_session(session_id = session_id)
    if result['status'] == u'SUCC':
        print_result('Async train result:', result)
        break
    time.sleep(1)
cam.release()
cv2.destroyAllWindows()
