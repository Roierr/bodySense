import numpy as np
import cv2
import glob
import os
import sys

# === CONFIGURACIÓN ===
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_FOTOS = os.path.join(CARPETA_BASE, "capturas")
ARCHIVO_FINAL = os.path.join(CARPETA_BASE, "stereo.npz")

CHECKERBOARD = (9, 6) 
TAMANO_CUADRO = 25
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

print(f"Leyendo fotos de: {CARPETA_FOTOS}")

if not os.path.exists(CARPETA_FOTOS):
    print("ERROR: No existe la carpeta de fotos.")
    sys.exit()

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * TAMANO_CUADRO

objpoints = [] 
imgpoints_l = [] 
imgpoints_r = [] 

def obtener_numero(nombre):
    try:
        base = os.path.basename(nombre)
        num = base.split('_')[-1].split('.')[0]
        return int(num)
    except: return -1

todos_archivos = os.listdir(CARPETA_FOTOS)
archivos_izq = sorted([f for f in todos_archivos if 'cam0' in f], key=obtener_numero)
archivos_der = sorted([f for f in todos_archivos if 'cam1' in f], key=obtener_numero)

pares_validos = []
for f_izq in archivos_izq:
    num = obtener_numero(f_izq)
    f_der_esperado = f"cam1_{num}.png"
    if f_der_esperado in archivos_der:
        pares_validos.append((os.path.join(CARPETA_FOTOS, f_izq), 
                              os.path.join(CARPETA_FOTOS, f_der_esperado)))

print(f"Pares encontrados: {len(pares_validos)}")
if len(pares_validos) < 10:
    print("ERROR: Muy pocas fotos. Toma al menos 15.")
    sys.exit()

img_shape = None

for img_l_path, img_r_path in pares_validos:
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if img_shape is None: img_shape = gray_l.shape[::-1]

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp)
        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
        imgpoints_l.append(corners2_l)
        imgpoints_r.append(corners2_r)
        print(f"  🔹 OK: {os.path.basename(img_l_path)}")

print("Calibrando... (Espera)")
ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)

flags = cv2.CALIB_FIX_INTRINSIC
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx1, dist1, mtx2, dist2, img_shape, 
    criteria=criteria, flags=flags)

print(f"Error RMS: {ret}")
np.savez(ARCHIVO_FINAL, mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, R=R, T=T)
print(f"Guardado: {ARCHIVO_FINAL}")


print("\nGenerando vista previa de rectificación...")

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, img_shape, R, T)
map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, img_shape, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, img_shape, cv2.CV_32FC1)

imgL = cv2.imread(pares_validos[0][0])
imgR = cv2.imread(pares_validos[0][1])

rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

total_width = rectL.shape[1] + rectR.shape[1]
height = rectL.shape[0]
canvas = np.zeros((height, total_width, 3), dtype=np.uint8)
canvas[:, :rectL.shape[1]] = rectL
canvas[:, rectL.shape[1]:] = rectR

for i in range(0, height, 30):
    cv2.line(canvas, (0, i), (total_width, i), (0, 255, 0), 1)

print("Abriendo ventana de validación (Presiona tecla para cerrar)")
cv2.imshow("RECTIFICACION (Lineas deben ser rectas)", cv2.resize(canvas, (1000, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()