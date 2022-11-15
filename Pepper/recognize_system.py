import cv2
import numpy as np
import tensorflow as tf
import eventlet
import socketio.server
import landmarks_connections as ddc

sio = socketio.Server()
app = socketio.WSGIApp(sio)
segni = np.array(['ciao', 'grazie', 'null', 'prego'])
model = tf.keras.models.load_model("model.h5")

@sio.event
def connect(sid, environ):
    print("conne", sid)

@sio.event
def disconnect(sid):
    print("disc", sid)

@sio.on('my_message')
def play_recognize(video):
    print("a")
    segni.sort()
    sequence = []
    predictions = []
    firstSegno = False
    lastSegno = ""

    cap = cv2.VideoCapture(video)

    with ddc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image, results = ddc.mediapipe_detection(frame, holistic)

            ddc.draw_styled_landmarks(image, results)
            keypoints = ddc.extract_keypoints(results)

            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(segni[np.argmax(res)])
                if(firstSegno == False):
                    lastSegno = segni[np.argmax(res)]
                    firstSegno = True
                if(lastSegno != segni[np.argmax(res)]):
                    #manda parola
                    lastSegno = segni[np.argmax(res)]
                predictions.append(np.argmax(res))

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 3000)), app)
    play_recognize()

