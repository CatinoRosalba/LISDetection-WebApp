import os
import numpy as np
import pprint as pp
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.backend import clear_session
from sklearn.metrics import accuracy_score

segni = np.array(['ciao', 'grazie', 'null', 'prego', 'amico', "mangiare", "bere"])
n_video = 75                                                                # numero DataSet1
frame_video = 30                                                            # ogni DataSet1 30 frame
log_dir = os.path.join('Logs')                                              # Log Directory
tb_callback = TensorBoard(log_dir=log_dir)                                  # info tensorflow

# assegna un nome ad ogni azione
def define_label():
    segni.sort()                         #ordine l'array per rispettare l'ordine alfabetico del dataset
    label_map = {label:num for num, label in enumerate(segni)}
    pp.pprint(label_map)
    keypointsDataSetPath = str(os.getcwd()) + "\keypointsDataset"
    keypointsList = os.listdir(keypointsDataSetPath)
    sequences, labels = [], []
    for keypointFolder in keypointsList:
        window = []
        for frame_num in range(frame_video):
            keypoint = np.load(os.path.join(keypointsDataSetPath, keypointFolder, "{}.npy".format(frame_num)))
            window.append(keypoint)
        sequences.append(window)
    for segno in segni:
        for i in range(n_video):
            labels.append(label_map[segno])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)  #to_categorical converte un vettore di interi in una matrice binaria
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Costruisce il modello della rete neurale
def create_model():
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, activation='tanh', input_shape=(30, 1662)))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(segni.shape[0], activation='softmax')) #3 neural units
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=150, callbacks=[tb_callback])
    pp.pprint(model.summary())
    return model

def make_predictions(model):
    res = model.predict(x_test)
    pp.pprint(segni[np.argmax(res[1])])
    pp.pprint(segni[np.argmax(y_test[1])])


def save_model(model):
    model.save('model.h5')
    pp.pprint("Model salvato")


def delete_model():
    clear_session()
    pp.pprint("Model cancellato")

# Calcola il livello di accuratezza della previsione in un range da 0 a 1
def evaluation_accuracy(model):
    yhat = model.predict(x_train)
    ytrue = np.argmax(y_train, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    print("Livello di accuratezza: ", accuracy_score(ytrue, yhat))




if __name__ == "__main__":
    define_label()
    delete_model()
    model = create_model()
    save_model(model)
    make_predictions(model)
    evaluation_accuracy(model)
    pp.pprint("Model creato")