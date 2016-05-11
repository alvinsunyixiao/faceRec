#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: hello.py

# In this tutorial, you will learn how to call Face ++ APIs and implement a
# simple App which could recognize a face image in 3 candidates.
# 在本教程中，您将了解到Face ++ API的基本调用方法，并实现一个简单的App，用以在3
# 张备选人脸图片中识别一个新的人脸图片。

# You need to register your App first, and enter you API key/secret.
# 您需要先注册一个App，并将得到的API key和API secret写在这里。
API_KEY = 'ae397874d1b74db82804dcfb67d7a249'
API_SECRET = 'kDH7U5ggdkDLPSLy27Tz-eXRZWJwTOKm'

# Import system libraries and define helper functions
# 导入系统库并定义辅助函数
import time
from pprint import pformat
import cv2
import numpy as np
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
import cv2


cam = cv2.VideoCapture(0)
api = API(API_KEY, API_SECRET)

# Here are the person names and their face images
# 人名及其脸部图片
PERSON = 'Kang'


api.person.create(person_name = PERSON, group_name = 'test')
while True:
    ret, img = cam.read()
    img = cv2.resize(img,(640,360))
    cv2.imwrite('buf.jpg',img)
    cv2.imshow('go',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key != 99:
        continue

    result = api.detection.detect(img = File('buf.jpg'), mode = 'oneface')
    print_result('Detection result for {}:'.format(PERSON), result)

    face_id = result['face'][0]['face_id']

    api.person.add_face(person_name = PERSON, face_id = face_id)


# Step 3: Train the group.
# Note: this step is required before performing recognition in this group,
# since our system needs to pre-compute models for these persons
# 步骤3：训练这个group
# 注：在group中进行识别之前必须执行该步骤，以便我们的系统能为这些person建模
result = api.recognition.train(group_name = 'test', type = 'all')

# Because the train process is time-consuming, the operation is done
# asynchronously, so only a session ID would be returned.
# 由于训练过程比较耗时，所以操作必须异步完成，因此只有session ID会被返回
print_result('Train result:', result)

session_id = result['session_id']

# Now, wait before train completes
# 等待训练完成
while True:
    result = api.info.get_session(session_id = session_id)
    if result['status'] == u'SUCC':
        print_result('Async train result:', result)
        break
    time.sleep(1)

cam.release()
cv2.destroyAllWindows()