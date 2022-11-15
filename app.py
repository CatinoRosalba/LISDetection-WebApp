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
app.config['UPLOAD_FOLDER_PIC'] = os.path.join('static', 'pic_alfabeto')

segni = np.array(['ciao', 'grazie', 'null', 'prego', 'amico', 'bere', 'mangiare', 'mamma', 'papà'])
alfabeto = np.array(['a', 'b', 'c', 'd', 'e'])
camera = cv2.VideoCapture(0)


def open_camera():
    while camera.isOpened():
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_segni():
    model = tf.keras.models.load_model("model_segni.h5")
    segni.sort()
    sequence = []
    last = ''
    with ddc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while not session["isRecognized"]:

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
                if not session["isRecognized"]:
                    if detected != last:
                        last = detected
                        yield "Riprova! "
                    if name_gif == detected:
                        session["isRecognized"] = True
                        session["counter"] = session.get("counter") + 1
                        yield "Corretto! "


def detect_alfabeto():
    model = tf.keras.models.load_model("model_alfabeto.h5")
    alfabeto.sort()
    sequence = []
    last = ''
    with ddc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while not session["isRecognized"]:

            ret, frame = camera.read()
            frame = cv2.flip(frame, 1)
            image, results = ddc.mediapipe_detection(frame, holistic)

            keypoints = ddc.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                detected = alfabeto[np.argmax(res)]
                print("detected: " + detected)
                print("pic: " + name_pic)
                if not session["isRecognized"]:
                    if detected != last:
                        last = detected
                        yield "Riprova! "
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


def randpic():
    pics = os.listdir(app.config['UPLOAD_FOLDER_PIC'])                      # salva il contenuto della cartella gif
    print(pics)
    pic_rand = random.choice(pics)                                          # sceglie una pic
    name_pic, ex_pic = pic_rand.split(".")                                  # splitta il nome
    path_picname = os.path.join(app.config['UPLOAD_FOLDER_PIC'], pic_rand)  # path della gif
    print(path_picname, name_pic)
    return path_picname, name_pic


# Index
@app.route('/index')
@app.route('/')
def index():
    session["counter"] = 0
    return Response(stream_with_context(render_template('index.html')))


# Pagina in cui viene mostrata la gif del segno da memorizzare
@app.route('/gif_segni')
def gif_segni():
    global path_gifname, name_gif
    path_gifname, name_gif = randgif()
    session["isRecognized"] = False
    return stream_template("gif_segni.html", sign_gif=path_gifname, name_gif=name_gif)


# Pagina minigioco
@app.route('/minigioco_segni')
def minigioco_segni():
    return stream_template("minigioco_segni.html", name_gif=name_gif)


@app.route('/pic_alfabeto')
def pic_alfabeto():
    global path_picname, name_pic
    path_picname, name_pic = randpic()
    session["isRecognized"] = False
    return stream_template("pic_alfabeto.html", alfabeto_pic=path_picname, name_pic=name_pic)


@app.route('/minigioco_alfabeto')
def minigioco_alfabeto():
    return stream_template("minigioco_alfabeto.html", name_pic=name_pic)


# In questo url viene eseguita solo la cam
@app.route('/video_feed')
def video_feed():
    return Response(open_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


# In questo url viene eseguita la detection del segno
@app.route('/detect_segno')
def return_detect_segno():
    return stream_with_context(detect_segni())


@app.route('/detect_alfabeto')
def return_detect_alfabeto():
    return stream_with_context(detect_alfabeto())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

