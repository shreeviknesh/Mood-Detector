import os
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import matplotlib.pyplot as plt

from collect_data import collect_training_data
from constants import *


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
    model.add(Conv2D(32, 3, input_shape=INPUT_SHAPE,
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))

    loss = 'sparse_categorical_crossentropy'
    optimizer = 'adam'  # use 'adam' also
    metrics = ['acc']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


if __name__ == "__main__":
    if not DATA_COLLECTED:
        collect_training_data(MOODS, TRAINING_FOLDER,
                              TRAIN_SIZE_PER_MOOD, IMAGE_SIZE)

    (train_x, train_y) = load_data()

    model = create_model(INPUT_SHAPE, NUM_CLASSES)
    hist = model.fit(train_x, train_y, epochs=EPOCHS,
                     batch_size=8, shuffle=True, validation_split=0.2)

    model.save('model/model.h5')
    del model

    plt.figure()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.plot(hist.history['loss'])
    plt.savefig('model/loss.png')
    plt.show()

    plt.figure()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy value')
    plt.plot(hist.history['acc'])
    plt.savefig('model/acc.png')
    plt.show()
