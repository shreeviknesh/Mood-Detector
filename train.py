from constants import *
from collect_data import collect_training_data
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
import pickle


def load_data():
    """A function to load the x,y values from the already created pickle file."""
    with open(os.path.join(os.path.abspath(TRAINING_FOLDER), 'training_data.pickle'), 'rb') as file:
        data = pickle.load(file)
    x = data['x']
    y = data['y']

    return (x, y)


def create_model(INPUT_SHAPE, NUM_CLASSES):
    """A function that creates a CNN model & returns it."""

    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=INPUT_SHAPE, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    if NUM_CLASSES == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    optimizer = 'adam'
    metrics = ['acc']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


class StopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.99:
            print(
                f'\nStopped training after epoch {epoch} because accuracy > 0.99.')
            self.model.stop_training = True


def fit_model(model, x, y, epochs, batch_size):
    hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[StopCallback()])

    if 'model' not in os.listdir(os.getcwd()):
        os.mkdir('model')
    model.save('model/model.h5')
    del model

    plt.figure()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.plot(hist.history['loss'])
    plt.savefig('model/loss.png')

    plt.figure()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy value')
    plt.plot(hist.history['acc'])
    plt.savefig('model/acc.png')


if __name__ == "__main__":
    if TRAINING_FOLDER in os.listdir(os.getcwd()):
        flag = input('training data seems to be present already, do you want to collect data again [Y/N]?')
        if flag.lower() == 'y':
            collect_training_data(MOODS, TRAINING_FOLDER, TRAIN_SIZE_PER_MOOD, IMAGE_SIZE)
    else:
        collect_training_data(MOODS, TRAINING_FOLDER, TRAIN_SIZE_PER_MOOD, IMAGE_SIZE)

    (train_x, train_y) = load_data()
    model = create_model(INPUT_SHAPE, NUM_CLASSES)
    fit_model(model, train_x, train_y, EPOCHS, 8)
