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
MIN_IDS_COMUNES = 4

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
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_l, shape_l, None, None)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_r, shape_r, None, None)

print("Calibrando estéreo...")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
flags = cv2.CALIB_FIX_INTRINSIC
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx1, dist1, mtx2, dist2, shape_l,
    criteria=criteria, flags=flags)

# Guardar archivo .npz
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
