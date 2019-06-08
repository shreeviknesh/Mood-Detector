import cv2


def detect_faces(img):
    face_cascade = cv2.CascadeClassifier(
        './haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Image', img)


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)

    while True:
        check, frame = capture.read()
        frame = cv2.flip(frame, 1)

        detect_faces(frame)
        key = cv2.waitKey(1)

        if key == ord(' '):
            break

    capture.release()
    cv2.destroyAllWindows()
