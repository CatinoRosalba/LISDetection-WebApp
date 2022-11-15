import socketio.client
import cv2
from naoqi import ALProxy, ALBroker
import vision_definitions

sio = socketio.Client()

@sio.event
def connect():
    print("connected")

@sio.event
def my_message(video):
    sio.emit(str(video))

@sio.event
def disconnect():
    print("disconnected")

def video():
    videoDevice = ALProxy('ALVideoDevice')
    AL_kTopCamera = vision_definitions.kTopCamera
    AL_kQVGA = vision_definitions.kQVGA
    AL_kBGRColorSpace = vision_definitions.kRGBColorSpace
    captureDevice = videoDevice.subscribeCamera("test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)
    captureDevice = cv2.VideoCapture(0)
    return captureDevice



if __name__ == "__main__":
    yBroker = ALBroker("myBroker", "0.0.0.0", 80, "192.168.191.245", 9559)
    sio.connect('http://127.0.0.1:3000/')
    video = video()
    my_message(video)