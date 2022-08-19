import cv2
cam = cv2.VideoCapture(0)
cam.set(3, 648)  # lebar cam
cam.set(4, 480)  # pnjg cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
id = input(
    'Masukkan Face Id yang akan direkam datanya [KEMUDIAN TEKAN ENTER] : ')
nama = input('Masukkan Nama : ')
print("Tatap wajah anda kedepan dalam webcam. Tunggu proses pengambilan data selesai...")
ambildata = 0
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(
        abuAbu, 1.3, 5)  # frame,scalefactor,minNeightbors
    for(x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imwrite('DataSet/'+str(nama)+'.'+str(id)+'.' +
                    str(ambildata)+'.jpg', abuAbu[y:y+h, x:x+w])
        ambildata += 1

        roi_gray = abuAbu[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detects eyes of different sizes in the input image
        eyes = eyeDetector.detectMultiScale(roi_gray)

        # To draw a rectangle in eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),
                          (ex+ew, ey+eh), (0, 0, 255), 1)

        # id, conf = recognizer.predict(abu[y:y+h, x:x+w])
    cv2.imshow('Kamera 1', frame)
    # cv2.imshow('Kamera 2', abuAbu)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif ambildata >= 30:
        break
print("pengambilan data selesai..")
cam.release()
cv2.destroyAllWindows()
