
import cv2
import numpy as np
from pathlib import Path

ARUCO_DICT   = cv2.aruco.DICT_4X4_50
MARKER_ID    = 0
MARKER_PX    = 400   
BORDER_PX    = 60    
OUTPUT_FILE  = Path(__file__).resolve().parent / "marcador_aruco_id0.png"

def main():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    marker_img = np.zeros((MARKER_PX, MARKER_PX), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, MARKER_ID, MARKER_PX, marker_img, 1)

    bordered = cv2.copyMakeBorder(
        marker_img,
        BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
        cv2.BORDER_CONSTANT,
        value=255
    )

    final = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

    info_h = 60
    canvas = np.ones((final.shape[0] + info_h, final.shape[1], 3), dtype=np.uint8) * 255
    canvas[:final.shape[0]] = final

    text = f"ArUco DICT_4X4_50 - ID {MARKER_ID} - Imprimir a 10 cm"
    cv2.putText(canvas, text, (10, final.shape[0] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)

    cv2.imwrite(str(OUTPUT_FILE), canvas)
    print(f"✅ Marcador guardado en: {OUTPUT_FILE}")
    print(f"   Tamaño total: {canvas.shape[1]}×{canvas.shape[0]} px")
    print(f"   Imprímelo y mide que el cuadro negro mida ~10 cm de lado.")

    # Mostrar preview
    cv2.imshow("Marcador ArUco ID=0  (cualquier tecla para cerrar)", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()