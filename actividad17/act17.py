import cv2
import numpy as np

def analizar_mascara_frutas():
    # AÑADE TU IMAGEN AQUÍ (Una foto de frutas con fondo neutro)
    img = cv2.imread('C:\\Users\\ekt\\OneDrive\\Desktop\\Graficacion\\actividad17\\frutas.png')
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mascara_bruta = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))

    kernel = np.ones((5, 5), np.uint8)
    mascara_limpia = cv2.morphologyEx(mascara_bruta, cv2.MORPH_OPEN, kernel)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)

    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_limpia, connectivity=8)

    frutas_validas = 0
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 500: 
            frutas_validas += 1
            print(f"Fruta #{frutas_validas} | Área: {area} px")

    print(f"\nTOTAL: {frutas_validas}")
    cv2.imshow("1. Original", img)
    cv2.imshow("2. Mascara Limpia", mascara_limpia)
    cv2.waitKey(0)

if __name__ == "__main__":
    analizar_mascara_frutas()