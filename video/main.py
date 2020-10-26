import cv2
import numpy as np

# cap = cv2.VideoCapture('http://192.168.0.108:8080/video')
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    print(ret)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
