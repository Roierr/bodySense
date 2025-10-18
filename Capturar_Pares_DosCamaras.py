import cv2
import numpy as np
import os


CHESSBOARD = (13, 9)
square_size = 18  


folder_left = os.path.abspath("calibracion_cam1")
folder_right = os.path.abspath("calibracion_cam2")
os.makedirs(folder_left, exist_ok=True)
os.makedirs(folder_right, exist_ok=True)
print("Carpeta izquierda:", folder_left)
print("Carpeta derecha:", folder_right)


cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("No se pudieron abrir las cámaras")
    exit()

count = 0

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not retL or not retR:
        print("Error al leer las cámaras")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    foundL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD, flags)
    foundR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD, flags)

    if foundL:
        cv2.drawChessboardCorners(frameL, CHESSBOARD, cornersL, foundL)
        cv2.putText(frameL, "Tablero detectado (Izquierda)", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(frameL, "Buscando tablero...", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    if foundR:
        cv2.drawChessboardCorners(frameR, CHESSBOARD, cornersR, foundR)
        cv2.putText(frameR, "Tablero detectado (Derecha)", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(frameR, "Buscando tablero...", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Camara Izquierda", frameL)
    cv2.imshow("Camara Derecha", frameR)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break


    if foundL and foundR:
        filenameL = os.path.join(folder_left, f"img_{count:03d}.jpg")
        filenameR = os.path.join(folder_right, f"img_{count:03d}.jpg")
        successL = cv2.imwrite(filenameL, frameL)
        successR = cv2.imwrite(filenameR, frameR)
        if successL and successR:
            count += 1
            print(f"Guardado par #{count}: {filenameL} + {filenameR}")
        else:
            print("Error al guardar imágenes. Revisa permisos y rutas.")

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
