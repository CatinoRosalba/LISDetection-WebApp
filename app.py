from flask import Flask, render_template, Response, stream_with_context, session, stream_template
import cv2
import os
import numpy as np
import tensorflow as tf
import random
import landmarks_connections as ddc

app = Flask(__name__)

# secret key aggiunta altrimenti dava errore il passaggio delle variaibli tramite session
app.config['SECRET_KEY'] = 'the random string'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'gif')

segni = np.array(['ciao', 'grazie', 'null', 'prego'])
model = tf.keras.models.load_model("model.h5")
camera = cv2.VideoCapture(0)

def open_camera():
    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect(name_gif):
    segni.sort()
    sequence = []
    last = ''
    with ddc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while session["isRecognized"] == False:

            ret, frame = camera.read()
            frame = cv2.flip(frame, 1)
            image, results = ddc.mediapipe_detection(frame, holistic)

            keypoints = ddc.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                detected = segni[np.argmax(res)]
                print("detected: " + detected)
                print("gif: " + name_gif)
                if session["isRecognized"] == False:
                    if detected != last:
                        last = detected
                        yield "Sbagliato "
                    if name_gif == detected:
                        session["isRecognized"] = True
                        session["counter"] = session.get("counter") + 1
                        yield "Corretto! "


def randgif():
    gifs = os.listdir(app.config['UPLOAD_FOLDER'])                      # salva il contenuto della cartella gif
    gif_rand = random.choice(gifs)                                      # sceglie una gif
    name_gif, ex_gif = gif_rand.split(".")                              # splitta il nome (es. "hello.gif" -> name_gif = "hello", ex_gif = "gif")
    path_gifname = os.path.join(app.config['UPLOAD_FOLDER'], gif_rand)  # path della gif
    return path_gifname, name_gif


#index
@app.route('/index')
@app.route('/')
def index():
    session["counter"] = 0
    return Response(stream_with_context(render_template('index.html')))


@app.route('/gif')
def gif():
    global path_gifname, name_gif
    path_gifname, name_gif = randgif()
    session["isRecognized"] = False
    return stream_template("gif.html", sign_gif=path_gifname, name_gif=name_gif)


#pagina minigioco
@app.route('/minigioco')
def minigioco():
    return stream_template("minigioco.html", name_gif=name_gif)


# In questo url viene eseguita solo la cam
@app.route('/video_feed')
def video_feed():
    return Response(open_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


# In questo url viene eseguita la detection
@app.route('/detect')
def detect_action():
    return stream_with_context(detect(name_gif))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)