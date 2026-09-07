import os
import sys

import cv2
import numpy as np

# Carpeta donde viven los archivos de calibracion (stereo.npz)
CARPETA_CALIBRACION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracion")

# Backend de captura: DirectShow solo existe en Windows. En macOS/Linux CAP_ANY
# deja que OpenCV elija el backend nativo (AVFoundation / V4L2).
BACKEND_CAMARA = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

# Tamano del tablero en ESQUINAS INTERNAS, no en cuadros: un tablero de
# 8x8 cuadros tiene 7x7 esquinas internas (donde se cruzan cuatro cuadros).
# Vive aqui porque capturar.py y generar_calibracion.py DEBEN usar el mismo:
# si difieren, se captura con un tamano y se calibra con otro.
# Medido con medir_tablero.py sobre el tablero real.
CHECKERBOARD = (7, 7)

# Flags de deteccion del tablero, los del ejemplo canonico de OpenCV.
# ADAPTIVE_THRESH umbraliza por regiones y NORMALIZE_IMAGE ecualiza el
# histograma antes de buscar. Ojo: OpenCV 4.x ya prueba varios umbrales por su
# cuenta, asi que con flags=0 detecta casi igual; estos estan explicitos por
# claridad, no porque arreglen un caso medido.
# capturar.py detecta para decidir si guarda la foto y generar_calibracion.py
# vuelve a detectar sobre esa misma foto, asi que ambos usan los mismos flags.
FLAGS_TABLERO = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def ordenar_esquinas(esquinas, tablero=CHECKERBOARD):
    """Reordena las esquinas detectadas a una orientacion canonica.

    findChessboardCorners numera las esquinas empezando por una que depende del
    angulo de vista. Un tablero CUADRADO (como 7x7) tiene cuatro orientaciones
    validas y un rectangular tiene dos, asi que dos camaras que miran el mismo
    tablero desde angulos distintos pueden numerarlo diferente. La calibracion
    estereo empareja la esquina i de una camara con la i de la otra, asi que ese
    desacuerdo cruza las correspondencias: las intrinsecas salen bien y el RMS
    estereo explota.

    Medido en este proyecto con un tablero 7x7: RMS estereo 12.79 px sin
    reordenar, 1.79 px con reordenado.

    Se elige la rotacion cuya primera esquina queda mas arriba-izquierda en la
    imagen, criterio que ambas camaras resuelven igual porque las dos ven el
    tablero aproximadamente derecho.
    """
    cols, filas = tablero
    rejilla = esquinas.reshape(filas, cols, 2)

    mejor, mejor_score = rejilla, None
    for giro in range(4):
        cand = np.rot90(rejilla, giro)
        if cand.shape[:2] != (filas, cols):
            continue  # un giro de 90 grados solo cabe si el tablero es cuadrado
        score = float(cand[0, 0, 0] + cand[0, 0, 1])
        if mejor_score is None or score < mejor_score:
            mejor, mejor_score = cand, score

    return np.ascontiguousarray(mejor.reshape(-1, 1, 2))


# Buscar archivo en múltiples ubicaciones, antes habia mas lugares y las rutas eran mas complicadas
# Se dejo por simplicidad y para evitar errores
def buscar_archivo_desesperadamente(nombre):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lugares = [
        os.path.join(base_dir, "calibracion"),
        os.path.join(base_dir, "data", "calibracion"),
        os.path.join(base_dir, "data"),
        base_dir,
        os.path.join(base_dir, ".."),
        os.path.join(base_dir, "..", "data", "calibracion"),
        os.getcwd()
    ]
    for ruta in lugares:
        path = os.path.join(ruta, nombre)
        if os.path.exists(path): return path
    for root, dirs, files in os.walk(base_dir):
        if nombre in files: return os.path.join(root, nombre)
    return None