import cv2
import numpy as np

mascara = cv2.imread('C:\\Users\\ekt\\OneDrive\\Desktop\\Graficacion\\actividad16\\calajo.png', cv2.IMREAD_UNCHANGED)  

if mascara is None:
    print("❌ ERROR: No encuentro la imagen de la máscara PNG (calajo.png).")
    exit()

ruta_xml = 'C:\\Users\\ekt\\OneDrive\\Desktop\\Graficacion\\actividad16\\haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(ruta_xml)

if face_cascade.empty():
    print("❌ ERROR: No encuentro el archivo XML en la carpeta actividad16.")
    exit()

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("❌ ERROR: No puedo acceder a la cámara.")
        break
        
    rostros = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

    for (x, y, w, h) in rostros:
        mascara_redimensionada = cv2.resize(mascara, (w, h))
        mascara_rgb = mascara_redimensionada[:, :, :3]
        mascara_alpha = mascara_redimensionada[:, :, 3]
        
        roi = frame[y:y+h, x:x+w]
        
        if roi.shape[:2] == mascara_alpha.shape[:2]:
            fondo = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mascara_alpha))
            mascara_fg = cv2.bitwise_and(mascara_rgb, mascara_rgb, mask=mascara_alpha)
            frame[y:y+h, x:x+w] = cv2.add(fondo, mascara_fg)

    cv2.imshow('Filtro Rostro', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

video.release()
cv2.destroyAllWindows()