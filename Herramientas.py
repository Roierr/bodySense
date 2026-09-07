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
# histograma antes de buscar. Aclaro: OpenCV 4.x ya prueba varios umbrales por su
# cuenta, asi que con flags=0 detecta casi igual; estos estan explicitos por
# claridad, no porque arreglen un caso medido.
# capturar.py detecta para decidir si guarda la foto y generar_calibracion.py
# vuelve a detectar sobre esa misma foto, asi que ambos usan los mismos flags.
FLAGS_TABLERO = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

# Refinacion subpixel para el detector clasico (SB ya devuelve subpixel).
CRITERIA_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# Fraccion minima del cuadro que debe ocupar el tablero para que capturar.py
# guarde la foto. Es un piso grueso, no una garantia: capturas al 9% dieron una
# calibracion inservible (focal de 1515 px, o sea 24 grados de campo de vision,
# implausible para una webcam). Lo que de verdad condiciona bien el sistema es
# VARIAR LA INCLINACION del tablero, no su tamano en el cuadro; el tamano solo
# ayuda a localizar mejor las esquinas. La verificacion real es a posteriori:
# que el fx resultante corresponda a un campo de vision creible.
# Es una perilla: si mi camara no logra enfocar el tablero tan cerca, la bajo a
# 0.10 en vez de quedarme sin capturar. El chequeo de campo de vision al final
# de generar_calibracion.py avisa si quedo mal condicionada.
MIN_ANCHO_TABLERO = 0.15


def tamano_relativo(esquinas, forma_img):
    """Que tan grande se ve el tablero, como fraccion del cuadro.

    Se toma el MAYOR de los dos spans (horizontal sobre el ancho, vertical
    sobre el alto), no solo el horizontal: inclinar el tablero sobre el eje
    vertical comprime su span en X aunque el tablero este igual de cerca.
    Midiendo solo en X, capturar.py pedia inclinar el tablero y a la vez
    castigaba haberlo inclinado: el tablero se detectaba perfecto y la barra
    seguia diciendo ACERCALO. Cada inclinacion conserva al menos uno de los
    dos ejes, asi que el mayor de los dos no se cae al inclinar.

    forma_img es frame.shape (alto, ancho, ...).
    """
    alto_img, ancho_img = forma_img[:2]
    p = esquinas.reshape(-1, 2)
    return max(float(np.ptp(p[:, 0])) / ancho_img,
               float(np.ptp(p[:, 1])) / alto_img)


def detectar_tablero(gris, tablero=CHECKERBOARD):
    """Detecta el tablero. Devuelve (encontrado, esquinas) ya en subpixel.

    Intenta primero findChessboardCornersSB, que localiza las esquinas con
    mejor precision subpixel que findChessboardCorners + cornerSubPix y no
    necesita margen blanco alrededor del tablero. Si SB no encuentra nada, cae
    al detector clasico + cornerSubPix, asi que solo puede detectar un
    superconjunto de lo que detectaba antes: ninguna captura que antes
    funcionaba se pierde.

    NOTA: con imagenes sinteticas (desenfoque, perspectiva fuerte, tablero sin
    margen) los dos detectores dieron el mismo resultado, asi que la ventaja de
    SB con imagenes reales aqui no esta medida, solo es la recomendada por
    OpenCV. El fallback es lo que hace seguro el cambio.

    capturar.py detecta para decidir si guarda la foto y generar_calibracion.py
    vuelve a detectar sobre esa misma foto: los dos pasan por aqui para que no
    puedan desincronizarse.
    """
    ok, esquinas = cv2.findChessboardCornersSB(gris, tablero, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        return True, esquinas

    ok, esquinas = cv2.findChessboardCorners(gris, tablero, FLAGS_TABLERO)
    if not ok:
        return False, None
    return True, cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), CRITERIA_SUBPIX)


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


# Busco el archivo en múltiples ubicaciones, antes habia mas lugares y las rutas eran mas complicadas
# Lo deje asi por simplicidad y para evitar errores
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