import cv2
import numpy as np
import glob
import os


CHESSBOARD = (13, 9)  
square_size = 18      

folder = r"C:\Users\royea\OneDrive\Escritorio\ProyectoDeImagenes\calibracionParaUnaCamara2"
images = glob.glob(os.path.join(folder, "*.jpg"))
print(f"Se encontraron {len(images)} imágenes.")

if len(images) == 0:
    print("Error: No se encontraron imágenes en la carpeta. Revisa la ruta y las extensiones.")
    exit()

objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  
imgpoints = []  


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    gray = cv2.equalizeHist(gray)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK


    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, flags)

    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHESSBOARD, corners2, found)
        print(f"Esquinas detectadas en: {os.path.basename(fname)}")

        cv2.imshow('Detección de esquinas', img)
        cv2.waitKey(200)
    else:
        print(f"No se detectaron esquinas en: {os.path.basename(fname)}")

cv2.destroyAllWindows()


print(f"\nSe detectaron esquinas en {len(objpoints)} imágenes.")
if len(objpoints) == 0:
    print("Error: No se detectaron esquinas en ninguna imagen. Revisa el tablero o la iluminación.")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n Calibración completada correctamente")
print("\nMatriz de cámara (mtx):\n", mtx)
print("\nCoeficientes de distorsión (dist):\n", dist)

np.savez("calibracion_cam2_params.npz", mtx=mtx, dist=dist)
print("Parámetros guardados en 'calibracion_cam2_params.npz'")
