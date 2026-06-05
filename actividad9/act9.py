import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cv2.waitKey(2000)
ret, background = cap.read() 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
    mask_inv = cv2.bitwise_not(mask)
    
    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    res2 = cv2.bitwise_and(background, background, mask=mask)
    
    cv2.imshow("Capa", cv2.addWeighted(res1, 1, res2, 1, 0))
    if cv2.waitKey(1) & 0xFF == ord('q'): break