import cv2
import numpy as np
import os
import actionDetection_helper as ddc

segni = np.array(['amico', 'mangiare', 'bere', 'grazie', 'prego', 'ciao', 'null'])                     # Azioni da riconoscere
n_video = 5                                                                         # Numero video per azione
frame_video = 30                                                                    # frame da salvare per ogni video

#crea le cartelle dove salvare se non esistono
def create_folders():
    os.chdir(str(os.getcwd()))
    if not os.path.exists('DataSet'):
        os.mkdir('DataSet')
    if not os.path.exists('keypointsDataset'):
        os.mkdir('keypointsDataset')

#permette all'utente di posizionarsi prima di registrare i video
def video_position():
    cap = cv2.VideoCapture(0)
    with ddc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = ddc.mediapipe_detection(frame, holistic)
            ddc.draw_styled_landmarks(image, results)
            image = cv2.flip(image, 1)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

#registra e salva i video
def register_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    for segno in segni:
        ready = False
        while ready == False:
            print(segno)
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, 'PREPARATI PER {} E PREMI Q'.format(segno), (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow('OpenCV .Feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                ready = True
        for video in range(n_video):
            filename = str(segno) + " " + "(" + str(video) + ")" + '.mp4'
            videopath = os.path.join('DataSet', filename)
            out = cv2.VideoWriter(videopath, fourcc, 30, (640, 480))
            for frame_num in range(frame_video+1):                  #perde un frame nel salvataggio e quindi ne registro 31
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if frame_num == 0:
                    cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(segno, video),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(2000)
                else:
                    cv2.imshow('OpenCV Feed', frame)
                    out.write(frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#estrae i keypoints dai video salvati
def extract_keypoints_dataset():
    videoDataSetPath = str(os.getcwd()) + "\DataSet"
    videoList = os.listdir(videoDataSetPath)
    for video in videoList:
        videoPath = os.path.join("DataSet", str(video))
        cap = cv2.VideoCapture(videoPath)
        print(video)
        videokeypath = os.path.join('keypointsDataset', str(video))
        os.mkdir(videokeypath)
        i_keypoints = 0
        with ddc.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for frame_num in range(frame_video):
                ret, frame = cap.read()
                image, results = ddc.mediapipe_detection(frame, holistic)
                ddc.draw_styled_landmarks(image, results)
                cv2.imshow('OpenCV Feed', frame)
                keypoints = ddc.extract_keypoints(results)
                keypointspath = os.path.join(videokeypath, str(i_keypoints))
                np.save(keypointspath, keypoints)
                i_keypoints = i_keypoints+1
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    create_folders()
    #video_position()
    #register_video()
    extract_keypoints_dataset()
