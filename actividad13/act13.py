import cv2 as cv
import numpy as np

img = cv.imread('C:\\Users\\ekt\\OneDrive\\Desktop\\Graficacion\\actividad13\\yopi.png', 1)

img2 = np.zeros((img.shape[:2]), dtype=np.uint8)
b, g, r = cv.split(img)

r2 = cv.merge([img2, img2, r]) 
g2 = cv.merge([img2, g, img2]) 

cv.imshow('Solo Rojo', r2)
cv.imshow('Solo Verde', g2)
cv.waitKey(0)
cv.destroyAllWindows()