import cv2

# Initializing some constants
MOODS = ['happy', 'sad']
TRAINING_FOLDER = 'Data/'
MODEL_FOLDER = 'Model/'
TRAIN_SIZE_PER_MOOD = 250
IMAGE_SIZE = (150, 150)

# One channel
INPUT_SHAPE = tuple(list(IMAGE_SIZE) + [1])
NUM_CLASSES = len(MOODS)
EPOCHS = 10

# The classifier that detects faces
FACE_CASCADE = cv2.CascadeClassifier('Vendor/opencv/haarcascade_frontalface_default.xml')