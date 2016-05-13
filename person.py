import cv2
from facepp import API,File
import time
API_KEY = 'ae397874d1b74db82804dcfb67d7a249'
API_SECRET = 'kDH7U5ggdkDLPSLy27Tz-eXRZWJwTOKm'

from pprint import pformat
import picamera

def capture_face(api,dir):
    result = api.detection.detect(img = File(dir), mode = 'oneface')
    if len(result['face'])==0:
        return None
    else:
        return result

def recgonize(cam, dir = 'buf.jpg'):
    api = API(API_KEY, API_SECRET)
    cam.capture(dir)
    result = api.recognition.identify(img = File(dir), group_name = 'test')
    if (len(result['face'])==0):
        return None
    firstFace = result['face'][0]['candidate'][0]
    name = firstFace['person_name']
    confidence = firstFace['confidence']
    return (name,confidence)

class Person:
    def __init__(self,name,bufferDir = 'buf.jpg'):
        self.name = name
        self.api = API(API_KEY, API_SECRET)
        self.bufferDir = bufferDir
        try:
            self.api.person.create(person_name = name, group_name = 'test')
        except:
            pass

    def print_result(self, hint, result):
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


    def train(self):
        cam = picamera.PiCamera()
	while True:
            cam.capture(self.bufferDir)
            result = capture_face(self.api,self.bufferDir)
            if result == None:
                continue
            face_id = result['face'][0]['face_id']
            self.api.person.add_face(person_name = self.name, face_id = face_id)
        cam.close()
        result = self.api.recognition.train(group_name = 'test', type = 'all')
        session_id = result['session_id']
        while True:
            result = self.api.info.get_session(session_id = session_id)
            if result['status'] == u'SUCC':
                self.print_result('Async train result:', result)
                break
            time.sleep(1)

