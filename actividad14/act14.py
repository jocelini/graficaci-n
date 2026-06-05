import cv2 as cv
import numpy as np

import cv2 as cv
import numpy as np

img = cv.imread('C:\\Users\\ekt\\OneDrive\\Desktop\\Graficacion\\actividad14\\rockandoleando.png', 0)
x, y = img.shape

M_tras = np.float32([[1, 0, 100], [0, 1, 50]])
img_tras = cv.warpAffine(img, M_tras, (y, x))
M_rot = cv.getRotationMatrix2D((y//2, x//2), 45, 1.0)
img_rot = cv.warpAffine(img, M_rot, (y, x))
img_esc = cv.resize(img, None, fx=2, fy=2)
img_espejo = cv.flip(img, 1)

cv.imshow('Rotada', img_rot)
cv.imshow('Espejo', img_espejo)
cv.waitKey(0)
x, y = img.shape

M_tras = np.float32([[1, 0, 100], [0, 1, 50]])
img_tras = cv.warpAffine(img, M_tras, (y, x))
M_rot = cv.getRotationMatrix2D((y//2, x//2), 45, 1.0)
img_rot = cv.warpAffine(img, M_rot, (y, x))
img_esc = cv.resize(img, None, fx=2, fy=2)
img_espejo = cv.flip(img, 1)

cv.imshow('Rotada', img_rot)
cv.imshow('Espejo', img_espejo)
cv.waitKey(0)