import cv2
import numpy as np

# cap = cv2.VideoCapture('http://192.168.0.108:8080/video')
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imwrite('test.png',frame)
