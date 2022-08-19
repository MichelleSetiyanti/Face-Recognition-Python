import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert(
            'L')  # convert ke dlm grey color
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids


print("Mesin Sedang melakukan Training.. tunggu dalam beberapa detik...")
faces, Ids = getImagesWithLabels('DataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('training/training.xml')
print("Data yang ditrainingkan ke mesin sebanyak :",
      format(len(np.unique(Ids))))
