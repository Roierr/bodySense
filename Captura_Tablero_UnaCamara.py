import cv2
import numpy as np
import os
import time

# --- Configurasao ---
CHESSBOARD = (13, 9)
folder = "calibracionParaUnaCamara2"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(1)  # Cámara
if not cap.isOpened():
    print("No se pudo abrir la camara")
    exit()

count = 0
ultimo_guardado = 0
intervalo = 1.0  # segundos entre guardados

print("Mueve el tablero frente a la camara.")
print("Presiona ESC para salir.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, flags)

    if found:
        # --- Dibujar las esquinas manualmente en naranja ---
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (int(x), int(y)), 6, (0, 140, 255), -1)  # BGR naranja brillante

        ahora = time.time()
        if ahora - ultimo_guardado > intervalo:
            filename = os.path.join(folder, f"img_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            ultimo_guardado = ahora
            print(f"Tablero detectado. Imagen guardada: {filename}")

        cv2.putText(frame, "Tablero detectado", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)  # texto naranja
    else:
        cv2.putText(frame, "Buscando tablero...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # rojo si no detecta

    # --- Mostrar ventana ---
    cv2.imshow("Captura automática", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captura terminada. Se guardaron {count} imágenes válidas en '{folder}'.")
