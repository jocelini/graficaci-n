

import cv2
import numpy as np

# cargar la imagen de Jake
imagen = cv2.imread('actividad1/jake.jpeg')

# verificar que la imagen se cargó
if imagen is None:
    print("Error: no se encontró la imagen 'jake.jpeg'")
    exit()

# escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# negativo
negativo = 255 - gris

# corrección gamma
gamma = 1.5
tabla = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
gamma_corrected = cv2.LUT(imagen, tabla)

# umbralización
_, umbral = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

# mostrar resultados
cv2.imshow('original', imagen)
cv2.imshow('grises', gris)
cv2.imshow('negativo', negativo)
cv2.imshow('gamma corregida', gamma_corrected)
cv2.imshow('umbralizacion', umbral)

cv2.waitKey(0)
cv2.destroyAllWindows()
