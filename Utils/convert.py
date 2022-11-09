import tensorflow as tf
import os
import numpy as np
import pprint as pp
from pathlib import Path
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.backend import clear_session
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


PATH_DIR = Path.cwd()
dataset_dir = PATH_DIR.joinpath('LIS')
saved_model_dir = PATH_DIR.joinpath('action')
saved_h5_dir = dataset_dir.joinpath('action.h5')



DATA_PATH = os.path.join('MP_Data')                                 # Path for exported data, numpy arrays
actions = np.array(['hello', 'thanks', 'iloveyou'])                 # Actions that we try to detect
no_sequences = 30                                                   # Thirty videos worth of data
sequence_length = 30                                                # Videos are going to be 30 frames in length
log_dir = os.path.join('Logs')                                      # Log Directory
tb_callback = TensorBoard(log_dir=log_dir)



#
# Crea un array di etichette che rappresentano le azioni
#
def define_label():
    label_map = {label:num for num, label in enumerate(actions)}
    pp.pprint(label_map)

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []         # singoli frame ottenuti dalla sequenza
            #lettaralmente "prendi il frame 0 e mettilo nell'array window etc.."
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            # Quando carica i 29 frame, li mette in coda nell'arrat sequences
            # in sequence sono contenuti i 90 (30*3 che sono le azioni)effettivi DataSet1
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)


    y = to_categorical(labels).astype(int) #to_categorical converte un vettore di interi in una matrice binaria


    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)



#
# Costruisce il modello della rete neurale
#
def create_model():
    model = Sequential()        # Istanzia
    # Rappresentano i livelli della rete neurale
    # Il primo numero (64, 32, 128) rappresentano i neroni nel livello Dense o LSTM
    # 'relu' converte l'output da un minimo di zero verso velori illimitati verso l'alto
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    # Qui è false perchè la linea successica è un Dense quindi non deve tornare alla sequenza di quel livello
    model.add(LSTM(64, return_sequences=False, activation='relu')) 
    model.add(Dense(64, activation='relu')) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax')) #3 neural units

    # Compile dice a tensorflow come vogliamo esercitare (train) il nostro model indicando metrica, ottimizzazione etc
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # è letteralmente il training così da poter fare delle previsioni più è il training più è precisa la previsione
    model.fit(x_train, y_train, epochs=500, callbacks=[tb_callback])
    
    pp.pprint(model.summary())

    return model


def make_predictions(model):
    res = model.predict(x_test)
    pp.pprint(actions[np.argmax(res[1])])
    pp.pprint(actions[np.argmax(y_test[1])])


def save_model(model):
    model.save('action')
    model.save('action.h5')
    pp.pprint("Model salvato")


def delete_model():
    clear_session()
    pp.pprint("Model cancellato")


#
# Calcola il livello di accuratezza della previsione in un range da 0 a 1 
#
def evaluation_accuraty(model):
    yhat = model.predict(x_train)
    ytrue = np.argmax(y_train, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    pp.pprint(multilabel_confusion_matrix(ytrue, yhat))
    print("Livello di accuratezza: ", accuracy_score(ytrue, yhat))


if __name__ == "__main__":
    define_label()
    model = create_model()
    make_predictions(model)
    save_model(model)
    evaluation_accuraty(model)

    num_calibration_steps = 1 # at least 100

    def representative_dataset_gen():
        for i in range(num_calibration_steps):
            # Remember to pre-process your dataset as your training
            imgs = x_train[i:i+1]
            imgs = imgs / 255
            yield [imgs.astype('float32')]


    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir.as_posix())

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8


    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    # save model
    tflite_model_file = PATH_DIR.joinpath('model_int8.tflite')
    tflite_model_file.write_bytes(tflite_model)