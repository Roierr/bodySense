#!/usr/bin/env python3
"""Genera calibracion estereo usando CharuCo boards.

Ejemplo de uso:
  python generar_calibracion_charuco.py

El script busca pares en la carpeta de capturas y procesa solo pares donde ambos
lados detectan suficientes marcadores comunes. Genera `stereo.npz` en la carpeta
de calibracion y un resumen JSON `stereo_report_charuco.json`.
"""

import json
import os
import sys
import numpy as np
import cv2

from utils.Herramientas import (CARPETA_CALIBRACION, CARPETA_CAPTURAS,
                                RANGO_FOV_CREIBLE, fov_grados,
                                crear_charuco, detectar_charuco, pares_de_capturas)

os.makedirs(CARPETA_CALIBRACION, exist_ok=True)
ARCHIVO_FINAL = os.path.join(CARPETA_CALIBRACION, "stereo.npz")
ARCHIVO_REPORTE = os.path.join(CARPETA_CALIBRACION, "stereo_report_charuco.json")

CUADROS_MINIMOS = 10
# 4 esquinas bastan para no caerse, pero la intrinseca queda inestable.
# 10+ esquinas por par es lo minimo para una focal creible.
MIN_IDS_COMUNES = 10

print(f"Leyendo fotos de: {CARPETA_CAPTURAS}")
if not os.path.exists(CARPETA_CAPTURAS):
    print("ERROR: no existe la carpeta de capturas")
    sys.exit(1)

pares = pares_de_capturas(CARPETA_CAPTURAS)
print(f"Pares encontrados: {len(pares)}")
if len(pares) < CUADROS_MINIMOS:
    print(f"ERROR: Se necesitan al menos {CUADROS_MINIMOS} pares.")
    sys.exit(1)

board, detector = crear_charuco()

objpoints = []
imgpoints_l = []
imgpoints_r = []
origins = []

shape_l = shape_r = None

for left, right in pares:
    img_l = cv2.imread(left)
    img_r = cv2.imread(right)
    if img_l is None or img_r is None:
        continue
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if shape_l is None:
        shape_l = (gray_l.shape[1], gray_l.shape[0])
        shape_r = (gray_r.shape[1], gray_r.shape[0])

    ok_l, corners_l, ids_l = detectar_charuco(gray_l, detector)
    ok_r, corners_r, ids_r = detectar_charuco(gray_r, detector)
    if not (ok_l and ok_r):
        continue

    comunes = np.intersect1d(ids_l, ids_r)
    if len(comunes) < MIN_IDS_COMUNES:
        continue

    # Obtener puntos objeto (2D en board) y convertir a 3D (z=0)
    obj2d = board.getChessboardCorners()[comunes].astype(np.float32)
    obj3d = np.hstack([obj2d, np.zeros((obj2d.shape[0], 1), dtype=np.float32)])

    def extract_imgpts(corners, ids, order):
        pts = []
        for ident in order:
            # corners shape (N,1,2)
            idx = int(np.where(ids == ident)[0][0])
            pt = corners[idx].reshape(2)
            pts.append(pt)
        return np.asarray(pts, dtype=np.float32)

    ip_l = extract_imgpts(corners_l, ids_l, comunes)
    ip_r = extract_imgpts(corners_r, ids_r, comunes)

    objpoints.append(obj3d)
    imgpoints_l.append(ip_l)
    imgpoints_r.append(ip_r)
    origins.append(os.path.basename(left))
    print(f"OK CHARUCO: {os.path.basename(left)}  ids={len(comunes)}")

if len(objpoints) < CUADROS_MINIMOS:
    print(f"ERROR: solo {len(objpoints)} pares validos. Captura mas fotos.")
    sys.exit(1)

print("Calibrando cámaras individualmente...")
if shape_l != shape_r:
    print(f"ERROR: las camaras capturan a tamanos distintos ({shape_l} vs {shape_r}).")
    print("Recaptura con la misma resolucion en ambas.")
    sys.exit(1)
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_l, shape_l, None, None)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_r, shape_r, None, None)

# Descarta pares con error alto, igual que generar_calibracion.py de ajedrez.
def _error_por_par(objp_lista, imgp_lista, rvecs, tvecs, mtx, dist):
    errores = []
    for k, op in enumerate(objp_lista):
        proy, _ = cv2.projectPoints(op, rvecs[k], tvecs[k], mtx, dist)
        errores.append(cv2.norm(imgp_lista[k], proy, cv2.NORM_L2) / len(op) ** 0.5)
    return np.array(errores)

err_l = _error_por_par(objpoints, imgpoints_l, rvecs1, tvecs1, mtx1, dist1)
err_r = _error_por_par(objpoints, imgpoints_r, rvecs2, tvecs2, mtx2, dist2)
peor = np.maximum(err_l, err_r)
umbral = max(1.5, 3.0 * float(np.median(peor)))
buenas = peor <= umbral
if buenas.sum() < len(peor):
    print(f"Descartando {(~buenas).sum()} de {len(peor)} pares con error alto (umbral {umbral:.2f} px)")
    if buenas.sum() < CUADROS_MINIMOS:
        print(f"ERROR: solo quedan {buenas.sum()} pares buenos. Vuelve a capturar.")
        sys.exit(1)
    objpoints = [objpoints[i] for i in range(len(buenas)) if buenas[i]]
    imgpoints_l = [imgpoints_l[i] for i in range(len(buenas)) if buenas[i]]
    imgpoints_r = [imgpoints_r[i] for i in range(len(buenas)) if buenas[i]]
    print(f"Recalibrando con {len(objpoints)} pares...")
    ret1, mtx1, dist1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, shape_l, None, None)
    ret2, mtx2, dist2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, shape_r, None, None)
print(f"RMS individual: cam0 = {ret1:.3f} px   cam1 = {ret2:.3f} px")

print("Calibrando estéreo...")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
flags = cv2.CALIB_FIX_INTRINSIC
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx1, dist1, mtx2, dist2, shape_l,
    criteria=criteria, flags=flags)

# Guardar archivo .npz solo si pasa el rechazo duro.
baseline = float(np.linalg.norm(T))
rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
motivos = []
if ret1 > 2.0 or ret2 > 2.0:
    motivos.append(f"RMS individual alto (cam0 {ret1:.2f} / cam1 {ret2:.2f}, max 2.0). "
                   "QUE HICISTE MAL: tablero movido o poca luz. Recaptura quieto.")
if ret > 1.5:
    motivos.append(f"RMS estereo alto ({ret:.2f}, max 1.5). "
                   "QUE HICISTE MAL: faltan angulos variados. Inclina el charuco en todas direcciones.")
for nombre, mtx, forma in (("cam0", mtx1, shape_l), ("cam1", mtx2, shape_r)):
    fov = fov_grados(mtx[0, 0], forma[0])
    bajo, alto = RANGO_FOV_CREIBLE
    if not (bajo < fov < alto):
        motivos.append(f"FOV {nombre} implausible ({fov:.0f}). "
                       "QUE HICISTE MAL: charuco muy chico o muy cerca.")
if not (50.0 <= baseline <= 1500.0):
    motivos.append(f"Separacion rara ({baseline:.0f} mm). QUE HICISTE MAL: pares no sincronizados.")
if rot > 10.0:
    motivos.append(f"Camaras muy giradas ({rot:.1f}, max 10). QUE HICISTE MAL: ponlas paralelas.")
if len(objpoints) < 15:
    motivos.append(f"Pocos pares ({len(objpoints)}, quiero 15+). QUE HICISTE MAL: charuco no visto en ambas.")
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

# Generar reporte JSON
report = {
    "pairs_used": len(objpoints),
    "rms_stereo": float(ret),
    "camera0_fx": float(mtx1[0,0]),
    "camera1_fx": float(mtx2[0,0]),
}
with open(ARCHIVO_REPORTE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"Reporte: {ARCHIVO_REPORTE}")

print("Verificando plausibilidad del FOV:")
for nombre, mtx, forma in (("cam0", mtx1, shape_l), ("cam1", mtx2, shape_r)):
    fov = fov_grados(mtx[0, 0], forma[0])
    bajo, alto = RANGO_FOV_CREIBLE
    veredicto = "creible" if bajo < fov < alto else "IMPLAUSIBLE - recaptura"
    print(f"  {nombre}: fx={mtx[0,0]:8.1f} -> campo de vision {fov:5.1f} grados   {veredicto}")

print("Hecho.")
