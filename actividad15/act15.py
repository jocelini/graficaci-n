import cv2
import numpy as np

img_gray = cv2.imread('C:\\Users\\ekt\\OneDrive\\Desktop\\Graficacion\\actividad15\\15.png', cv2.IMREAD_GRAYSCALE)

_, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
filas, cols = binary_img.shape
visitado = np.zeros((filas, cols), dtype=bool)
output_img = binary_img.copy()
vecinos_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
conteo = 0

for i in range(filas):
    for j in range(cols):
        if not visitado[i, j] and binary_img[i, j] == 255:
            conteo += 1
            pila = [(i, j)]
            visitado[i, j] = True
            while pila:
                x, y = pila.pop()
                output_img[x, y] = 90
                for dx, dy in vecinos_8:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < filas and 0 <= ny < cols and binary_img[nx, ny] == 255 and not visitado[nx, ny]:
                        visitado[nx, ny] = True
                        pila.append((nx, ny))

print(f"Manchas: {conteo}")
cv2.imshow("Resultado", output_img)
cv2.waitKey(0)