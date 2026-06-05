

import cv2
import numpy as np

def aplicar_transformaciones():
    # Cargar imagen y obtener dimensiones
    foto = cv2.imread("actividad5/appab.jpeg")

    if foto is None:
        print("Error: No se encontro appab.jpeg")
        return

    h, w = foto.shape[:2]  # alto y ancho

    # ── Rotación 30° ──────────────────────────────────────────────────────
    img_rotada = np.zeros_like(foto)
    theta = np.radians(30)          # conversión a radianes
    c, s  = np.cos(theta), np.sin(theta)
    mid_x, mid_y = w // 2, h // 2  # pivote al centro

    for i in range(h):
        for j in range(w):
            # Coordenadas relativas al centro
            x_rel = j - mid_x
            y_rel = i - mid_y

            new_x = int(x_rel * c - y_rel * s + mid_x)
            new_y = int(x_rel * s + y_rel * c + mid_y)

            # Verificar que el nuevo píxel sigue dentro del lienzo
            if 0 <= new_x < w and 0 <= new_y < h:
                img_rotada[new_y, new_x] = foto[i, j]

    # ── Escalado 0.7x ─────────────────────────────────────────────────────
    img_escalada = np.zeros((h, w, 3), dtype=np.uint8)
    factor_x, factor_y = 0.7, 0.7

    for i in range(h):
        for j in range(w):
            # Qué píxel de la original corresponde a esta posición
            orig_x = int(j / factor_x)
            orig_y = int(i / factor_y)

            if 0 <= orig_x < w and 0 <= orig_y < h:
                img_escalada[i, j] = foto[orig_y, orig_x]

    # ── Traslación (80, 50) ───────────────────────────────────────────────
    img_trasladada = np.zeros_like(foto)
    offset_x, offset_y = 80, 50    # desplazamiento en píxeles

    for i in range(h):
        for j in range(w):
            target_x = j + offset_x
            target_y = i + offset_y

            if 0 <= target_x < w and 0 <= target_y < h:
                img_trasladada[target_y, target_x] = foto[i, j]

    # ── Reflexión vertical ────────────────────────────────────────────────
    img_reflejo = cv2.flip(foto, 0)  # 0 = reflexión vertical

    # ── Mostrar resultados ────────────────────────────────────────────────
    cv2.imshow("Original",               foto)
    cv2.imshow("rotacion 30 deg",     img_rotada)
    cv2.imshow("escalado 0.7x",       img_escalada)
    cv2.imshow("traslacion (80, 50)", img_trasladada)
    cv2.imshow("reflejo vertical",    img_reflejo)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    aplicar_transformaciones()