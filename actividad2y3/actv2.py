

import numpy as np
import cv2


#  ACTIVIDAD 2 — Pixel Art del Alien 


# Tamaño de cada celda en píxeles
S = 15

# ── Paleta de colores BGR ──────────────────────────────────────────────────
BLANCO = [255, 255, 255]   # B — fondo blanco
VERDE  = [30,  220, 180]   # V — cuerpo verde lima del alien
CIAN   = [220, 200,  30]   # C — lágrimas cian
NEGRO  = [20,   20,  20]   # N — contorno negro
SOMBRA = [60,  130,  80]   # S — verde oscuro / sombras
GRIS   = [160, 160, 160]   # G — gris contorno suave

# ── Mapa del alien 40×40 ──────────────────────────────────────────────────
# Cada letra = un color de la paleta
# Extraído directamente de la imagen de referencia
MAPA = [
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",  # F00
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",  # F01
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",  # F02
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",  # F03
    "BBBBBBBBBBBBBBBBBBBGGGGBBBBBBBBBBBBBBBBB",  # F04
    "BBBBBBBBBBBBBBBBBGGNSSSSGBBBBBBBBBBBBBBB",  # F05
    "BBBBBBBBBBBBBBBGNSVVVVVVSSGBBBBBBBBBBBBB",  # F06
    "BBBBBBBBBBBBBGGSVVVVVVVVVVVNBBBBBBBBBBBB",  # F07
    "BBBBBBBBBBBBGSSVVVVVVVVVVVVVGBBBBBBBBBBB",  # F08
    "BBBBBBBBBBBGSVVVVVVVVVVVVVVVGGBBBBBBBBBB",  # F09
    "BBBBBBBBBBBGVVVVVVVVVVVVVVVVVSGBBBBBBBBB",  # F10
    "BBBBBBBBBBGSVVVVVVVVVVVVVVVVVVNBBBBBBBBB",  # F11
    "BBBBBBBBBGSVVVVVVVVVVVVVVVVVVVSGBBBBBBBB",  # F12
    "BBBBBBBBBSVVVVVVVVVVVVVVVVVVVVGGBBBBBBBB",  # F13
    "BBBBBBBBGSVVVVVVVVVVVVVVVVVVVVGGBBBBBBBB",  # F14
    "BBBBBBBBGVVVVVVVVVVVVVVVVVVVVVGGBBBBBBBB",  # F15
    "BBBBBBBBSVVVVVVVVVVVVVVGGGVVVVGGBBBBBBBB",  # F16
    "BBBBBBBGSGSGVVVVVVVVVGGNNGGVVVGGBBBBBBBB",  # F17
    "BBBBBBBGNNGNSVVVVVVVGGNGBGNSVVGGBBBBBBBB",  # F18
    "BBBBBBBGNGNNNVVVVVVVGNGGGNNNVVGGBBBBBBBB",  # F19
    "BBBBBBBGNNNNNVVVVVVVGSSSSVVSSVGGBBBBBBBB",  # F20
    "BBBBBBBGNSGSSVVVVVVVVVVVVVVGGGGGBBBBBBBB",  # F21
    "BBBBBBGGCCCCVVVVVVVVVVVVGCCCCCCGBBBBBBBB",  # F22
    "BBBBBGCCCCCGVVVVVVVVVVVVGCCCCCCGBBBBBBBB",  # F23
    "BBBBBBGGGGVVVVVVVVVVVVVVVVVGGGBBBBBBBBBB",  # F24
    "BBBBBBBBBSVVVVVVVVVVVVGVVVGGGBBBBBBBBBBB",  # F25
    "BBBBBBBBBGSGSSVVGVVVVGGSVGNGBBBBBBBBBBBB",  # F26
    "BBBBBBBBBBGSGSSGNGNGGSSGGNGBBBBBBBBBBBBB",  # F27
    "BBBBBBBBBBBGNGGNGNNNSGGGGBBBBBBBBBBBBBBB",  # F28
    "BBBBBBBBBBBBGNNSGSGNGNGGBBBBBBBBBBBBBBBB",  # F29
    "BBBBBBBBBBBBBBSGNNVSNGBBBBBBBBBBBBBBBBBB",  # F30
    "BBBBBBBBBBBBBGSGSSVGGBBBBBBBBBBBBBBBBBBB",  # F31
    "BBBBBBBBBBBBBGSVVVVGGBBBBBBBBBBBBBBBBBBB",  # F32
    "BBBBBBBBBBBBGNVVVVVGGBBBBBBBBBBBBBBBBBBB",  # F33
    "BBBBBBBBBBBGSVGGGVVGGBBBBBBBBBBBBBBBBBBB",  # F34
    "BBBBBBBBBBBGSGGNGSVGGBBBBBBBBBBBBBBBBBBB",  # F35
    "BBBBBBBBBBBBGGGBBSGGBBBBBBBBBBBBBBBBBBBB",  # F36
    "BBBBBBBBBBBBBBBBBGNGBBBBBBBBBBBBBBBBBBBB",  # F37
    "BBBBBBBBBBBBBBBBBBGBBBBBBBBBBBBBBBBBBBBB",  # F38
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",  # F39
]

COLOR_MAP = {
    'B': BLANCO,
    'V': VERDE,
    'C': CIAN,
    'N': NEGRO,
    'S': SOMBRA,
    'G': GRIS,
}

# ── Crear imagen vacía con np.zeros (igual que el profe) ──────────────────
FILAS = len(MAPA)
COLS  = len(MAPA[0])
img   = np.zeros((FILAS * S, COLS * S, 3), dtype=np.uint8)

# ── Pintar cada celda con slicing ─────────────────────────────────────────
for f, fila in enumerate(MAPA):
    for c, letra in enumerate(fila):
        color = COLOR_MAP[letra]
        img[f*S:(f+1)*S, c*S:(c+1)*S] = color

# Guardar imagen original
cv2.imwrite("alien_pixelart.png", img)
print("[OK] alien_pixelart.png guardada")


#  ACTIVIDAD 3 — 5 Operadores Puntuales


# -- Operador 1: Negativo -------------------------------------------------------
# s = 255 - r  →  invierte cada valor de intensidad
op1 = 255 - img
cv2.imwrite("alien_op1_negativo.png", op1)
print("[OK] alien_op1_negativo.png guardada")

# -- Operador 2: Aumento de brillo -----------------------------------------------
# s = clip(r + 80, 0, 255)  →  suma constante con saturación
op2 = np.clip(img.astype(np.int32) + 80, 0, 255).astype(np.uint8)
cv2.imwrite("alien_op2_brillo.png", op2)
print("[OK] alien_op2_brillo.png guardada")

# -- Operador 3: Umbralización --------------------------------------------------
# Convierte a gris y binariza con umbral T = 127
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, op3_bin = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
op3 = cv2.cvtColor(op3_bin, cv2.COLOR_GRAY2BGR)
cv2.imwrite("alien_op3_umbral.png", op3)
print("[OK] alien_op3_umbral.png guardada")

# -- Operador 4: Corrección Gamma -----------------------------------------------
# s = (r/255)^gamma * 255  →  gamma = 0.5 aclara la imagen
tabla_gamma = np.array(
    [((i / 255.0) ** 0.5) * 255 for i in range(256)],
    dtype=np.uint8
)
op4 = cv2.LUT(img, tabla_gamma)
cv2.imwrite("alien_op4_gamma.png", op4)
print("[OK] alien_op4_gamma.png guardada")

# -- Operador 5: Escala de grises -----------------------------------------------
# BGR → GRAY → BGR  (para visualizar en ventana de 3 canales)
op5 = cv2.cvtColor(
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    cv2.COLOR_GRAY2BGR
)
cv2.imwrite("alien_op5_grises.png", op5)
print("[OK] alien_op5_grises.png guardada")

# ── Mostrar todas las imágenes ────────────────────────────────────────────
cv2.imshow("Original —Pixel Art", img)
cv2.imshow("negativo",             op1)
cv2.imshow("brillo +80",           op2)
cv2.imshow("umbral 127",           op3)
cv2.imshow("gamma 0.5",            op4)
cv2.imshow("escala de grises",     op5)

print("\nPresiona cualquier tecla para cerrar las ventanas...")
cv2.waitKey(0)
cv2.destroyAllWindows()