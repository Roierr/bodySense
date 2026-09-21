import numpy as np
import cv2
import os
import sys

from utils.Herramientas import (CARPETA_CALIBRACION, CARPETA_CAPTURAS, CHECKERBOARD,
                          RANGO_FOV_CREIBLE, TAMANO_CUADRO, detectar_tablero,
                          fov_grados, ordenar_esquinas, pares_de_capturas,
                          crear_charuco, detectar_charuco, puntos_de_charuco)

CARPETA_FOTOS = CARPETA_CAPTURAS
os.makedirs(CARPETA_CALIBRACION, exist_ok=True)
ARCHIVO_FINAL = os.path.join(CARPETA_CALIBRACION, "stereo.npz")

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

pares_validos = pares_de_capturas(CARPETA_FOTOS)

print(f"Pares encontrados: {len(pares_validos)}")
if len(pares_validos) < 10:
    print("ERROR: Muy pocas fotos. Toma al menos 15.")
    sys.exit()

shape_l = shape_r = None
origen = []


for img_l_path, img_r_path in pares_validos:
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if shape_l is None:
        shape_l = gray_l.shape[::-1]
        shape_r = gray_r.shape[::-1]

    # Modo: checkerboard (por defecto) o charuco
    modo = "checkerboard"
    if len(sys.argv) > 1 and sys.argv[1].lower().startswith("charu"):
        modo = "charuco"

    if modo == "charuco":
        # preparar board/detector
        board, detector = crear_charuco()
        ok_l, corners_l, ids_l = detectar_charuco(gray_l, detector)
        ok_r, corners_r, ids_r = detectar_charuco(gray_r, detector)
        if ok_l and ok_r:
            # buscar ids comunes
            comunes = np.intersect1d(ids_l, ids_r)
            if len(comunes) >= 10:
                # obtener puntos 3D en el orden de 'comunes'
                pts_obj = puntos_de_charuco(board, comunes)
                if pts_obj.ndim == 2:
                    pts_obj3 = np.hstack([pts_obj, np.zeros((pts_obj.shape[0], 1), dtype=np.float32)])
                else:
                    pts_obj3 = pts_obj

                def _extraer(corners, ids, orden):
                    pts = []
                    for ident in orden:
                        idx = int(np.where(ids == ident)[0][0])
                        pt = corners[idx].reshape(2)
                        pts.append(pt)
                    return np.asarray(pts, dtype=np.float32)

                ip_l = _extraer(corners_l, ids_l, comunes)
                ip_r = _extraer(corners_r, ids_r, comunes)

                objpoints.append(pts_obj3)
                imgpoints_l.append(ip_l)
                imgpoints_r.append(ip_r)
                origen.append(os.path.basename(img_l_path))
                print(f"OK CHARUCO: {os.path.basename(img_l_path)}  ids={len(comunes)}")
    else:
        ret_l, corners_l = detectar_tablero(gray_l)
        ret_r, corners_r = detectar_tablero(gray_r)

        if ret_l and ret_r:
            objpoints.append(objp)
            imgpoints_l.append(ordenar_esquinas(corners_l))
            imgpoints_r.append(ordenar_esquinas(corners_r))
            origen.append(os.path.basename(img_l_path))
            print(f"OK: {os.path.basename(img_l_path)}")

if len(objpoints) < 10:
    print(f"\nERROR: solo {len(objpoints)} de {len(pares_validos)} pares tienen el "
          f"tablero visible en AMBAS camaras. Se necesitan al menos 10.")
    sys.exit()

print("Calibrando... (Espera)")
print(f"Tamano cam0: {shape_l[0]}x{shape_l[1]}   cam1: {shape_r[0]}x{shape_r[1]}")
ret1, mtx1, dist1, rvec1, tvec1 = cv2.calibrateCamera(objpoints, imgpoints_l, shape_l, None, None)
ret2, mtx2, dist2, rvec2, tvec2 = cv2.calibrateCamera(objpoints, imgpoints_r, shape_r, None, None)

def error_por_par(objp_lista, imgp_lista, rvecs, tvecs, mtx, dist):
    errores = []
    for k, op in enumerate(objp_lista):
        proy, _ = cv2.projectPoints(op, rvecs[k], tvecs[k], mtx, dist)
        errores.append(cv2.norm(imgp_lista[k], proy, cv2.NORM_L2) / len(op) ** 0.5)
    return np.array(errores)


err_l = error_por_par(objpoints, imgpoints_l, rvec1, tvec1, mtx1, dist1)
err_r = error_por_par(objpoints, imgpoints_r, rvec2, tvec2, mtx2, dist2)
peor = np.maximum(err_l, err_r)
umbral = max(1.5, 3.0 * float(np.median(peor)))
buenas = peor <= umbral

if buenas.sum() < len(peor):
    descartadas = [i for i, ok in enumerate(buenas) if not ok]
    print(f"\nDescartando {len(descartadas)} de {len(peor)} fotos con error alto "
          f"(umbral {umbral:.2f} px):")
    for i in descartadas:
        print(f"  {origen[i]}  error {peor[i]:.2f} px")

    if buenas.sum() < 10:
        print(f"\nERROR: solo quedan {buenas.sum()} fotos buenas. Vuelve a capturar.")
        sys.exit()

    objpoints = [objpoints[i] for i in range(len(buenas)) if buenas[i]]
    imgpoints_l = [imgpoints_l[i] for i in range(len(buenas)) if buenas[i]]
    imgpoints_r = [imgpoints_r[i] for i in range(len(buenas)) if buenas[i]]
    print(f"Recalibrando con {len(objpoints)} fotos...")
    ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, shape_l, None, None)
    ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, shape_r, None, None)

print(f"RMS individual: cam0 = {ret1:.3f} px   cam1 = {ret2:.3f} px")

flags = cv2.CALIB_FIX_INTRINSIC
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx1, dist1, mtx2, dist2, shape_l,
    criteria=criteria, flags=flags)

print(f"\nError RMS estereo: {ret:.3f} px  " +
      ("(bien)" if ret < 1.5 else "(ALTO: recaptura)" if ret > 2.5 else "(aceptable)"))

print("\nVerificacion de plausibilidad:")
for nombre, mtx, forma in (("cam0", mtx1, shape_l), ("cam1", mtx2, shape_r)):
    fov = fov_grados(mtx[0, 0], forma[0])
    bajo, alto = RANGO_FOV_CREIBLE
    veredicto = "creible" if bajo < fov < alto else "IMPLAUSIBLE - recaptura"
    print(f"  {nombre}: fx={mtx[0,0]:8.1f} -> campo de vision {fov:5.1f} grados   {veredicto}")

baseline = float(np.linalg.norm(T))
rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
print(f"\nGeometria: separacion {baseline:.1f} mm, rotacion entre camaras {rot:.1f} grados")

# Rechazo duro: no se tolera calibracion muy distorsionada.
motivos = []
if ret1 > 2.0 or ret2 > 2.0:
    motivos.append(f"RMS individual alto (cam0 {ret1:.2f} / cam1 {ret2:.2f}, max 2.0). "
                   "QUE HICISTE MAL: fotos borrosas o tablero en movimiento. "
                   "Recaptura quieto y con buena luz.")
if ret > 1.5:
    motivos.append(f"RMS estereo alto ({ret:.2f}, max 1.5). "
                   "QUE HICISTE MAL: pocas inclinaciones distintas del tablero. "
                   "Inclinalo arriba/abajo/izq/der y acerca/aleja.")
for nombre, mtx, forma in (("cam0", mtx1, shape_l), ("cam1", mtx2, shape_r)):
    fov = fov_grados(mtx[0, 0], forma[0])
    bajo, alto = RANGO_FOV_CREIBLE
    if not (bajo < fov < alto):
        motivos.append(f"FOV {nombre} implausible ({fov:.0f} grados). "
                       "QUE HICISTE MAL: tablero muy chico en cuadro o muy cerca. "
                       "Llenalo al 30-60% y recaptura.")
if not (50.0 <= baseline <= 1500.0):
    motivos.append(f"Separacion rara ({baseline:.0f} mm, espero 50-1500). "
                   "QUE HICISTE MAL: las fotos no son pares sincronizados. Recaptura.")
if rot > 10.0:
    motivos.append(f"Camaras muy giradas ({rot:.1f} grados, max 10). "
                   "QUE HICISTE MAL: no estan paralelas mirando al frente. Endereza.")
if len(objpoints) < 15:
    motivos.append(f"Pocos pares validos ({len(objpoints)}, quiero 15+). "
                   "QUE HICISTE MAL: el tablero no se vio en ambas a la vez. Captura mas.")
if motivos:
    txt = "CALIBRACION RECHAZADA:\n" + "\n".join(f"- {m}" for m in motivos)
    print("\n" + txt)
    try:
        with open(os.path.join(CARPETA_CALIBRACION, "ultimo_rechazo.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    except OSError:
        pass
    sys.exit(2)
try:
    os.remove(os.path.join(CARPETA_CALIBRACION, "ultimo_rechazo.txt"))
except OSError:
    pass

np.savez(ARCHIVO_FINAL, mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, R=R, T=T)
print(f"Guardado: {ARCHIVO_FINAL}")

if shape_l != shape_r:
    print(f"\nVista previa omitida: las camaras capturan a tamanos distintos "
          f"({shape_l[0]}x{shape_l[1]} vs {shape_r[0]}x{shape_r[1]}).")
    print("La calibracion si quedo guardada y main.py la puede usar.")
    sys.exit()

print("\nGenerando vista previa de rectificación...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, shape_l, R, T)
map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, shape_l, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, shape_r, cv2.CV_32FC1)

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
