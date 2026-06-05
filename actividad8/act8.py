import cv2
import numpy as np

camara = cv2.VideoCapture(0)
ret, cuadro = camara.read()
lienzo = np.zeros_like(cuadro)
u_bajo, u_alto = np.array([0, 150, 50]), np.array([10, 255, 255])
punto_anterior = None

while True:
    ret, cuadro = camara.read()
    cuadro = cv2.flip(cuadro, 1)
    hsv = cv2.cvtColor(cuadro, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, u_bajo, u_alto)
    momentos = cv2.moments(mascara)
    
    if momentos["m00"] > 0:
        cx, cy = int(momentos["m10"] / momentos["m00"]), int(momentos["m01"] / momentos["m00"])
        punto_actual = (cx, cy)
        cv2.circle(cuadro, punto_actual, 5, (0, 0, 255), -1)
        if punto_anterior is not None and np.linalg.norm(np.array(punto_actual) - np.array(punto_anterior)) < 50:
            cv2.line(lienzo, punto_anterior, punto_actual, (0, 255, 0), 5)
        punto_anterior = punto_actual
    else:
        punto_anterior = None
        
    cv2.imshow("Dibujo", cv2.add(cuadro, lienzo))
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
    elif k == ord('c'): lienzo = np.zeros_like(cuadro)