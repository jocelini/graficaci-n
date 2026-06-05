import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while True:
    ret, imagen = cap.read()
    if not ret: break
    
    mascara_azul = cv.inRange(cv.cvtColor(imagen, cv.COLOR_BGR2HSV), np.array([100, 40, 40]), np.array([140, 255, 255]))
    imagen_gris_bgr = cv.cvtColor(cv.cvtColor(imagen, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    
    resultado = np.where(mascara_azul[:, :, None] == 255, imagen, imagen_gris_bgr)
    cv.imshow('Filtro Gris', resultado)
    if cv.waitKey(1) & 0xFF == 27: break