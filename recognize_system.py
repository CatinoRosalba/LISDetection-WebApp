import cv2
import numpy as np
import tensorflow as tf
import landmarks_connections as ddc

segni = np.array(['ciao', 'grazie', 'null', 'prego'])
model = tf.keras.models.load_model("model.h5")

def play_recognize():
    segni.sort()
    sequence = []

    cap = cv2.VideoCapture(0)

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

            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    play_recognize()