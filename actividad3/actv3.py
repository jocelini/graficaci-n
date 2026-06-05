import cv2 as cv  

# Cargar la imagen 'rojo.png' en color
img = cv.imread('rojo.png', 1)  # 1 indica que se carga en color BGR

# Convertir la imagen de BGR a HSV (Hue, Saturation, Value)
# Esto facilita la detección de colores específicos
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Definir los rangos de color rojo en HSV
# Primer rango de rojo (bajo)
uba = (10, 255, 255)  # Valor máximo del rango superior
ubb = (0, 60, 60)     # Valor mínimo del rango inferior

# Segundo rango de rojo (alto)
uba2 = (180, 255, 255)
ubb2 = (172, 60, 60)

# Crear máscaras para detectar el rojo en cada rango
mask1 = cv.inRange(hsv, ubb, uba)     # Máscara para el primer rango de rojo
mask2 = cv.inRange(hsv, ubb2, uba2)   # Máscara para el segundo rango de rojo

# Combinar ambas máscaras para capturar todo el rojo
mask = mask1 + mask2

# Aplicar la máscara a la imagen original
# Esto deja solo las partes rojas visibles
res = cv.bitwise_and(img, img, mask=mask)

# Mostrar resultados en ventanas separadas
cv.imshow('mask1', mask1)  # Primer rango de rojo
cv.imshow('mask2', mask2)  # Segundo rango de rojo
cv.imshow('mask', mask)    # Máscara combinada
cv.imshow('res', res)      # Resultado final con solo las áreas rojas visibles

# Esperar a que el usuario presione cualquier tecla
cv.waitKey(0)

# Cerrar todas las ventanas abiertas
cv.destroyAllWindows()