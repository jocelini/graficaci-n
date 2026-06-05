"""
Transformaciones Geométricas
"""

import cv2 as cv
import numpy as np

# Cargar la imagen en escala de grises
img = cv.imread('actividad4/appab.jpeg', 0)

if img is None:
    print("Error: No se encontro appab.jpeg")
    exit()

# Obtener el tamaño de la imagen
x, y = img.shape

# Definir el desplazamiento en x e y
dx, dy = 100, 50

# Crear la matriz de traslación
M = np.float32([[1, 0, dx], [0, 1, dy]])

# Aplicar la traslación usando warpAffine
# rellena los cuadros vacíos al mover la imagen mediante multiplicación matricial
translated_img = cv.warpAffine(img, M, (y, x))

# Mostrar la imagen original y la trasladada
cv.imshow('Original', img)
cv.imshow('Traslacion (100, 50)', translated_img)

cv.waitKey(0)
cv.destroyAllWindows()