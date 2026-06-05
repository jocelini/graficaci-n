import cv2
import numpy as np

ancho, alto = 600, 400
pos_x, pos_y, vel_x, vel_y, radio = 100, 100, 5, 3, 20

while True:
    fondo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    cv2.circle(fondo, (pos_x, pos_y), radio, (0, 255, 0), -1)
    
    pos_x += vel_x
    pos_y += vel_y
    if pos_x - radio <= 0 or pos_x + radio >= ancho: vel_x = -vel_x
    if pos_y - radio <= 0 or pos_y + radio >= alto: vel_y = -vel_y
        
    cv2.imshow("Bolita", fondo)
    if cv2.waitKey(30) & 0xFF == 27: break
cv2.destroyAllWindows()