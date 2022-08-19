import cv2
cam = cv2.VideoCapture(0)
while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Kamera 1', frame)
    cv2.imshow('Kamera 2', abuAbu)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break
cam.release()
cv2.destroyAllWindows()
