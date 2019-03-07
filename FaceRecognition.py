import cv2 as cv
import numpy as np

import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('/Library/Python/2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/Library/Python/2.7/site-packages/cv2/data/haarcascade_eye.xml')

video_capture = cv.VideoCapture(0)
video_capture.set(10,200)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.9,  # scaleFactor=1.1, #default value
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        mugshot = frame[y:y + h, x:x + w]
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv.imshow('mugshot', mugshot) # The mugshot has a green boarder?

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()