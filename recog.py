from tkinter import font
import cv2
import os
import numpy as np

wajahDir = 'DataSet'
trainingDir = 'training'
cam = cv2.VideoCapture(0)
cam.set(3, 648)  # lebar cam
cam.set(4, 480)  # pnjg cam

faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read('training/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Diketahui', 'Michelle', 'Very', 'Kartini']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    # id += 1
    retV, frame = cam.read()
    # frame = cv2.flip(frame, 1)
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceDetector.detectMultiScale(
        abuAbu, 1.3, 5, minSize=(round(minWidth), round(minHeight)),)
    # frame,scalefactor,minNeightbors
    for(x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = faceRecognizer.predict(
            abuAbu[y:y+h, x:x+w])  # confidence = artinya  cocok
        # print(id)
        if confidence <= 50:
            nameID = names[id]
            confidenceTXT = " {0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTXT = " {0}%".format(round(100-confidence))
        cv2.putText(frame, str(nameID), (x+5, y-5),
                    font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidenceTXT), (x+5, y+h-5),
                    font, 1, (255, 255, 0), 1)
    cv2.imshow('WebcamKu', frame)
    # cv2.imshow('Kamera 2', abuAbu)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
print("EXIT")
cam.release()
cv2.destroyAllWindows()
