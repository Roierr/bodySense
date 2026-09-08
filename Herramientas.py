import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde viven los archivos de calibracion (stereo.npz)
CARPETA_CALIBRACION = os.path.join(BASE, "calibracion")
# Carpeta donde capturar.py guarda los pares de fotos del tablero
CARPETA_CAPTURAS = os.path.join(BASE, "capturas")
# Config editable desde inicio.py
ARCHIVO_CONFIG = os.path.join(BASE, "config.json")

# Backend de captura: DirectShow solo existe en Windows. En macOS/Linux CAP_ANY
# deja que OpenCV elija el backend nativo (AVFoundation / V4L2).
BACKEND_CAMARA = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

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


# ==========================================
# CONFIG DEL TABLERO Y LA CAPTURA
# ==========================================
# Estos cinco valores vivian como constantes en el codigo. Ahora salen de
# config.json para que inicio.py los edite sin abrir el editor. El default de
# cada uno es el valor que tenia antes, asi que sin config.json el proyecto se
# comporta igual que siempre.
#
# checkerboard son ESQUINAS INTERNAS, no cuadros: un tablero de 8x8 cuadros
# tiene 7x7 esquinas internas (donde se cruzan cuatro cuadros). Medilo con
# medir_tablero.py. capturar.py y generar_calibracion.py DEBEN usar el mismo:
# si difieren, se captura con un tamano y se calibra con otro.
#
# tamano_cuadro_mm es el lado de un cuadro EN MILIMETROS, medido con regla. Es
# el unico dato del mundo real que entra al sistema y fija la escala metrica de
# TODA la reconstruccion. Si esta mal, la calibracion converge igual, el RMS
# sale igual de bueno, y todas las distancias 3D quedan mal por ese factor. Es
# el error mas dificil de detectar del proyecto, y la razon principal por la
# que vale la pena poder editarlo desde el menu.
#
# min_ancho_tablero es la fraccion minima del cuadro que debe ocupar el tablero
# para que capturar.py guarde la foto. Es un piso grueso, no una garantia:
# capturas al 9% dieron una calibracion inservible (focal de 1515 px, o sea 24
# grados de campo de vision, implausible para una webcam). Lo que de verdad
# condiciona bien el sistema es VARIAR LA INCLINACION del tablero, no su tamano
# en el cuadro. Es una perilla: si la camara no logra enfocar tan cerca, bajarla
# a 0.10 es mejor que quedarse sin capturar.
DEFAULTS_CONFIG = {
    "checkerboard": [7, 7],
    "tamano_cuadro_mm": 20.0,
    "min_ancho_tablero": 0.15,
    "total_fotos": 30,
    "intervalo_segundos": 2.0,
}

# Rangos plausibles. Un valor fuera de rango NO se usa: se cae al default y se
# avisa. config.json lo escribe un menu y lo puede editar una persona a mano,
# asi que es una frontera de confianza: un cero en checkerboard o un tamano de
# cuadro negativo produciria una calibracion basura sin ningun error visible.
LIMITES_CONFIG = {
    "tamano_cuadro_mm": (1.0, 500.0),
    "min_ancho_tablero": (0.02, 0.90),
    "total_fotos": (10, 200),
    "intervalo_segundos": (0.2, 30.0),
}


def validar_config(crudo):
    """Devuelve (config limpia, lista de avisos). Nunca lanza excepcion."""
    limpio, avisos = dict(DEFAULTS_CONFIG), []

    cb = crudo.get("checkerboard")
    if cb is not None:
        if (isinstance(cb, (list, tuple)) and len(cb) == 2
                and all(isinstance(v, int) and not isinstance(v, bool)
                        and 3 <= v <= 20 for v in cb)):
            limpio["checkerboard"] = [int(cb[0]), int(cb[1])]
        else:
            avisos.append(f"checkerboard {cb!r} invalido "
                          f"(se esperan dos enteros de 3 a 20); usando "
                          f"{DEFAULTS_CONFIG['checkerboard']}")

    for clave, (bajo, alto) in LIMITES_CONFIG.items():
        v = crudo.get(clave)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            avisos.append(f"{clave} {v!r} no es un numero; usando "
                          f"{DEFAULTS_CONFIG[clave]}")
        elif not bajo <= v <= alto:
            avisos.append(f"{clave} {v} fuera del rango [{bajo}, {alto}]; "
                          f"usando {DEFAULTS_CONFIG[clave]}")
        else:
            limpio[clave] = int(v) if clave == "total_fotos" else float(v)

    return limpio, avisos


def cargar_config():
    """Lee config.json. Si falta o esta roto, devuelve los defaults."""
    try:
        with open(ARCHIVO_CONFIG, encoding="utf-8") as f:
            crudo = json.load(f)
        if not isinstance(crudo, dict):
            raise ValueError("el contenido no es un objeto JSON")
    except FileNotFoundError:
        return dict(DEFAULTS_CONFIG), []
    except (json.JSONDecodeError, ValueError, OSError, UnicodeDecodeError) as e:
        return dict(DEFAULTS_CONFIG), [f"config.json ilegible ({e})"]
    return validar_config(crudo)


def guardar_config(cambios):
    """Mezcla cambios sobre la config actual, valida y escribe config.json.

    Devuelve (config guardada, avisos). Valida ANTES de escribir para que el
    menu no pueda dejar en disco un valor que despues se ignore en silencio.

    Relee el disco en vez de mezclar sobre CONFIG, que quedo congelada en el
    import: mezclando sobre esa, editar el tablero y despues el tamano de
    cuadro revertia el tablero.
    """
    actual, _ = cargar_config()
    limpio, avisos = validar_config({**actual, **cambios})
    with open(ARCHIVO_CONFIG, "w", encoding="utf-8") as f:
        json.dump(limpio, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return limpio, avisos


CONFIG, AVISOS_CONFIG = cargar_config()
for _aviso in AVISOS_CONFIG:
    print(f"AVISO config.json: {_aviso}")

CHECKERBOARD = tuple(CONFIG["checkerboard"])
TAMANO_CUADRO = CONFIG["tamano_cuadro_mm"]
MIN_ANCHO_TABLERO = CONFIG["min_ancho_tablero"]
TOTAL_FOTOS = CONFIG["total_fotos"]
INTERVALO_SEGUNDOS = CONFIG["intervalo_segundos"]


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


# Rango de campo de vision horizontal creible para una camara web, en grados.
RANGO_FOV_CREIBLE = (40.0, 130.0)


def fov_grados(fx, ancho_img):
    """Campo de vision horizontal, en grados, que implica una focal en pixeles.

    Es el chequeo que delata una calibracion mal condicionada aunque su RMS se
    vea bien: una focal de 1515 px sobre un cuadro de 640 son 24 grados, y
    ninguna webcam tiene ese lente. generar_calibracion.py lo imprime al
    terminar y inicio.py lo muestra siempre; los dos usan el mismo rango para
    no discrepar sobre que calibracion es creible.
    """
    return 2 * np.degrees(np.arctan(ancho_img / (2 * fx)))


def pares_de_capturas(carpeta=CARPETA_CAPTURAS):
    """Pares (foto_cam0, foto_cam1) que existen para el MISMO indice, ordenados.

    Vive aqui porque inicio.py muestra cuantos pares hay y
    generar_calibracion.py decide con cuantos calibra: si cada uno los contara
    a su manera, el menu diria 18 y el calibrador 12.

    Se exige la extension .png en las dos: si no, un cam0_3.jpg suelto se
    emparejaria con un cam1_3.png de otra sesion.
    """
    try:
        archivos = set(os.listdir(carpeta))
    except OSError:
        return []

    pares = []
    for nombre in archivos:
        if not (nombre.startswith("cam0_") and nombre.endswith(".png")):
            continue
        try:
            n = int(nombre[len("cam0_"):-len(".png")])
        except ValueError:
            continue
        pareja = f"cam1_{n}.png"
        if pareja in archivos:
            pares.append((n, os.path.join(carpeta, nombre),
                          os.path.join(carpeta, pareja)))

    return [(izq, der) for _, izq, der in sorted(pares)]


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