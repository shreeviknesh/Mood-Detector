import os
import pickle

from collect_data import collect_training_data

# Initializing some constants
MOODS = ['happy', 'sad']
TRAINING_FOLDER = 'training_data'
TRAIN_SIZE_PER_MOOD = 100
IMAGE_SIZE = (150, 150)

# 1 Channel
INPUT_SHAPE = tuple(list(IMAGE_SIZE) + [1])
NUM_CLASSES = len(MOODS)

# A variable to flag if the data has been collected or not
DATA_COLLECTED = False


def load_data():
    """A function to load the x,y values from the already created pickle file."""
    with open(os.path.join(os.path.abspath(TRAINING_FOLDER), 'training_data.pickle'), 'rb') as file:
        data = pickle.load(file)
    x = data['x']
    y = data['y']

    return (x, y)


if __name__ == "__main__":
    if not DATA_COLLECTED:
        collect_training_data(MOODS, TRAINING_FOLDER,
                              TRAIN_SIZE_PER_MOOD, IMAGE_SIZE)

    (train_x, train_y) = load_data()
