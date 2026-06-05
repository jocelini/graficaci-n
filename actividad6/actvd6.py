import cv2 as cv

cap = cv.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret: break
    
    img_gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    cv.imshow('Original', img)
    cv.imshow('Gris', img_gris)
    cv.imshow('HSV', hsv)
    
    if cv.waitKey(1) & 0xFF == 27: break
cap.release()
cv.destroyAllWindows()