import numpy as np
import cv2
import os
import sys

from Herramientas import (CARPETA_CALIBRACION, CHECKERBOARD, detectar_tablero,
                          ordenar_esquinas)

# === CONFIGURACIÓN ===
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_FOTOS = os.path.join(CARPETA_BASE, "capturas")
# Guardo donde main.py ya busca la calibracion, no en la raiz
os.makedirs(CARPETA_CALIBRACION, exist_ok=True)
ARCHIVO_FINAL = os.path.join(CARPETA_CALIBRACION, "stereo.npz")

# Parámetros del tablero de ajedrez. CHECKERBOARD viene de Herramientas.py
# para que capturar.py y este archivo no puedan desincronizarse.
# TAMANO_CUADRO es el lado de un cuadro EN MILIMETROS, medido con regla: es el
# unico dato del mundo real que entra al sistema y fija la escala metrica de
# TODA la reconstruccion. Si esta mal, la calibracion converge igual, el RMS
# sale igual de bueno, y todas las distancias 3D quedan mal por ese factor.
TAMANO_CUADRO = 20
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001) # Criterios para la refinación de esquinas

print(f"Leyendo fotos de: {CARPETA_FOTOS}") 

# Verifico que la carpeta de fotos exista
if not os.path.exists(CARPETA_FOTOS):
    print("ERROR: No existe la carpeta de fotos.")
    sys.exit()

# Preparo los puntos 3D del tablero de ajedrez
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * TAMANO_CUADRO

# Listas para puntos 3D y puntos 2D de ambas cámaras
objpoints = [] 
imgpoints_l = [] 
imgpoints_r = [] 

# Función para extraer el número de la imagen, OSEA EL ÍNDICE de la foto

def obtener_numero(nombre):
    try:
        base = os.path.basename(nombre)
        num = base.split('_')[-1].split('.')[0]
        return int(num)
    except: return -1

# Obtengo las listas de archivos de ambas cámaras

todos_archivos = os.listdir(CARPETA_FOTOS)
archivos_izq = sorted([f for f in todos_archivos if 'cam0' in f], key=obtener_numero)
archivos_der = sorted([f for f in todos_archivos if 'cam1' in f], key=obtener_numero)

# Emparejo los archivos por número de indice con la función obtener_numero
pares_validos = []
for f_izq in archivos_izq:
    num = obtener_numero(f_izq)
    f_der_esperado = f"cam1_{num}.png"
    if f_der_esperado in archivos_der:
        pares_validos.append((os.path.join(CARPETA_FOTOS, f_izq), 
                              os.path.join(CARPETA_FOTOS, f_der_esperado)))

# Muestro la cantidad de pares encontrados
print(f"Pares encontrados: {len(pares_validos)}")
if len(pares_validos) < 10:
    print("ERROR: Muy pocas fotos. Toma al menos 15.")
    sys.exit()

    # Proceso cada par de imágenes

# Cada camara tiene su propio tamano; no se asume que sean iguales.
shape_l = shape_r = None
# De que foto vino cada entrada de objpoints, para poder nombrarla si se descarta.
origen = []


for img_l_path, img_r_path in pares_validos:
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if shape_l is None:
        shape_l = gray_l.shape[::-1]
        shape_r = gray_r.shape[::-1]

    # detectar_tablero ya devuelve las esquinas refinadas a subpixel.
    ret_l, corners_l = detectar_tablero(gray_l)
    ret_r, corners_r = detectar_tablero(gray_r)

# Si se encuentran las esquinas en ambas imágenes, las agrego a la lista
    if ret_l and ret_r:
        objpoints.append(objp)
        # Sin esto, las dos camaras pueden numerar el mismo tablero empezando
        # por esquinas distintas y la calibracion estereo queda cruzada.
        imgpoints_l.append(ordenar_esquinas(corners_l))
        imgpoints_r.append(ordenar_esquinas(corners_r))
        origen.append(os.path.basename(img_l_path))
        print(f"OK: {os.path.basename(img_l_path)}")

# Sin tableros detectados, calibrateCamera muere con "nimages > 0", que no
# dice nada del problema real. La causa casi siempre es una de estas tres.
if len(objpoints) < 10:
    print(f"\nERROR: solo {len(objpoints)} de {len(pares_validos)} pares tienen el "
          f"tablero visible en AMBAS camaras. Se necesitan al menos 10.")
    print(f"El tablero configurado es CHECKERBOARD = {CHECKERBOARD}, o sea "
          f"{CHECKERBOARD[0]+1}x{CHECKERBOARD[1]+1} cuadros. Revisa:")
    print("  1. Que coincida con tu tablero fisico (corre medir_tablero.py)")
    print("  2. Que capturas/ no tenga fotos viejas de otro tablero (borra la carpeta)")
    print("  3. Que el tablero se vea completo y con luz en las dos camaras")
    sys.exit()

# Calibro las cámaras individualmente y luego en estéreo

print("Calibrando... (Espera)")
print(f"Tamano cam0: {shape_l[0]}x{shape_l[1]}   cam1: {shape_r[0]}x{shape_r[1]}")
ret1, mtx1, dist1, rvec1, tvec1 = cv2.calibrateCamera(objpoints, imgpoints_l, shape_l, None, None)
ret2, mtx2, dist2, rvec2, tvec2 = cv2.calibrateCamera(objpoints, imgpoints_r, shape_r, None, None)


def error_por_par(objp_lista, imgp_lista, rvecs, tvecs, mtx, dist):
    """Error de reproyeccion de cada foto, en pixeles."""
    errores = []
    for k, op in enumerate(objp_lista):
        proy, _ = cv2.projectPoints(op, rvecs[k], tvecs[k], mtx, dist)
        errores.append(cv2.norm(imgp_lista[k], proy, cv2.NORM_L2) / len(op) ** 0.5)
    return np.array(errores)


# Unas pocas fotos con movimiento o un angulo extremo pueden arruinar toda la
# calibracion: se detectan igual, pero sus esquinas quedan mal localizadas.
# Se descartan las que se salen del comportamiento del resto (mediana x3).
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

# Calibración estéreo. Con CALIB_FIX_INTRINSIC las intrinsecas ya calculadas
# arriba no se tocan, solo se resuelve R y T, asi que el imageSize casi no pesa.
flags = cv2.CALIB_FIX_INTRINSIC
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx1, dist1, mtx2, dist2, shape_l,
    criteria=criteria, flags=flags)

# Muestro los resultados
print(f"\nError RMS estereo: {ret:.3f} px  " +
      ("(bien)" if ret < 1.5 else "(ALTO: recaptura)" if ret > 2.5 else "(aceptable)"))

# Chequeo de plausibilidad: la focal en pixeles implica un campo de vision.
# Si sale fuera del rango de una camara real, la calibracion esta mal
# condicionada aunque el RMS se vea bien. Es el error que no avisa solo.
print("\nVerificacion de plausibilidad:")
for nombre, mtx, forma in (("cam0", mtx1, shape_l), ("cam1", mtx2, shape_r)):
    fov = 2 * np.degrees(np.arctan(forma[0] / (2 * mtx[0, 0])))
    veredicto = "creible" if 40 < fov < 130 else "IMPLAUSIBLE - recaptura"
    print(f"  {nombre}: fx={mtx[0,0]:8.1f} -> campo de vision {fov:5.1f} grados   {veredicto}")

baseline = float(np.linalg.norm(T))
rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
print(f"\nGeometria: separacion {baseline:.1f} mm, rotacion entre camaras {rot:.1f} grados")

np.savez(ARCHIVO_FINAL, mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2, R=R, T=T)
print(f"Guardado: {ARCHIVO_FINAL}")

# Genero la vista previa de rectificación
# stereoRectify recibe un solo imageSize, asi que la vista previa solo tiene
# sentido si las dos camaras capturan igual. La calibracion de arriba ya quedo
# guardada y es valida con tamanos distintos; esto es solo la validacion visual.
if shape_l != shape_r:
    print(f"\nVista previa omitida: las camaras capturan a tamanos distintos "
          f"({shape_l[0]}x{shape_l[1]} vs {shape_r[0]}x{shape_r[1]}).")
    print("La calibracion si quedo guardada y main.py la puede usar.")
    sys.exit()

print("\nGenerando vista previa de rectificación...")
# Calculo los mapas de rectificación

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, shape_l, R, T)
map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, shape_l, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, shape_r, cv2.CV_32FC1)

# Aplico la rectificación a un par de imágenes de ejemplo
imgL = cv2.imread(pares_validos[0][0])
imgR = cv2.imread(pares_validos[0][1])

# Aplico el remapeo
rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# Combino las imágenes para visualización
total_width = rectL.shape[1] + rectR.shape[1]
height = rectL.shape[0]
canvas = np.zeros((height, total_width, 3), dtype=np.uint8)
canvas[:, :rectL.shape[1]] = rectL
canvas[:, rectL.shape[1]:] = rectR

# Dibujo líneas horizontales para verificar la rectificación, esto es opcional pero sirve para ver si la calibracion fue buena
for i in range(0, height, 30):
    cv2.line(canvas, (0, i), (total_width, i), (0, 255, 0), 1)

# Muestro la imagen combinada con líneas horizontales
print("Abriendo ventana de validación (Presiona tecla para cerrar)")
cv2.imshow("RECTIFICACION (Lineas deben ser rectas)", cv2.resize(canvas, (1000, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()