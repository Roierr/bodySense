import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde viven los archivos de calibracion (stereo.npz)
CARPETA_CALIBRACION = os.path.join(BASE, "..", "calibracion")
# Carpeta donde capturar.py guarda los pares de fotos del tablero
CARPETA_CAPTURAS = os.path.join(BASE, "..", "capturas")
# Config editable desde inicio.py
ARCHIVO_CONFIG = os.path.join(BASE, "..", "config.json")

# Backend de captura: DirectShow solo existe en Windows. En macOS/Linux CAP_ANY
# deja que OpenCV elija el backend nativo (AVFoundation / V4L2).
BACKEND_CAMARA = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

# Flags de deteccion del tablero, los del ejemplo canonico de OpenCV.
FLAGS_TABLERO = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

CRITERIA_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

DEFAULTS_CONFIG = {
    "checkerboard": [7, 7],
    "tamano_cuadro_mm": 20.0,
    "min_ancho_tablero": 0.15,
    "total_fotos": 30,
    "intervalo_segundos": 2.0,
    "tablero_plegado": False,
    "tablero_tipo": "ajedrez",
    "charuco_cuadros": [5, 7],
    "charuco_cuadro_mm": 30.0,
    "charuco_marcador_mm": 22.0,
    "charuco_diccionario": "DICT_5X5_100",
}

BOOLEANOS_CONFIG = ("tablero_plegado",)
TIPOS_TABLERO = ("ajedrez", "charuco")
LIMITES_CONFIG = {
    "tamano_cuadro_mm": (1.0, 500.0),
    "min_ancho_tablero": (0.02, 0.90),
    "total_fotos": (10, 200),
    "intervalo_segundos": (0.2, 30.0),
    "charuco_cuadro_mm": (1.0, 500.0),
    "charuco_marcador_mm": (1.0, 500.0),
}


def _pareja_de_enteros(v, bajo, alto):
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, int) and not isinstance(x, bool)
                    and bajo <= x <= alto for x in v))


def validar_config(crudo):
    limpio, avisos = dict(DEFAULTS_CONFIG), []

    cb = crudo.get("checkerboard")
    if cb is not None:
        if _pareja_de_enteros(cb, 3, 20):
            limpio["checkerboard"] = [int(cb[0]), int(cb[1])]
        else:
            avisos.append(f"checkerboard {cb!r} invalido "
                          f"(se esperan dos enteros de 3 a 20); usando "
                          f"{DEFAULTS_CONFIG['checkerboard']}")

    cc = crudo.get("charuco_cuadros")
    if cc is not None:
        if _pareja_de_enteros(cc, 3, 30):
            limpio["charuco_cuadros"] = [int(cc[0]), int(cc[1])]
        else:
            avisos.append(f"charuco_cuadros {cc!r} invalido "
                          f"(se esperan dos enteros de 3 a 30); usando "
                          f"{DEFAULTS_CONFIG['charuco_cuadros']}")

    tt = crudo.get("tablero_tipo")
    if tt is not None:
        if tt in TIPOS_TABLERO:
            limpio["tablero_tipo"] = tt
        else:
            avisos.append(f"tablero_tipo {tt!r} desconocido "
                          f"(se espera uno de {TIPOS_TABLERO}); usando "
                          f"{DEFAULTS_CONFIG['tablero_tipo']}")

    dic = crudo.get("charuco_diccionario")
    if dic is not None:
        if isinstance(dic, str) and dic.startswith("DICT_") and hasattr(cv2.aruco, dic):
            limpio["charuco_diccionario"] = dic
        else:
            avisos.append(f"charuco_diccionario {dic!r} no existe en esta version "
                          f"de OpenCV; usando {DEFAULTS_CONFIG['charuco_diccionario']}")

    for clave in BOOLEANOS_CONFIG:
        v = crudo.get(clave)
        if v is None:
            continue
        if isinstance(v, bool):
            limpio[clave] = v
        else:
            avisos.append(f"{clave} {v!r} no es true ni false; usando "
                          f"{DEFAULTS_CONFIG[clave]}")

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

    if limpio["charuco_marcador_mm"] >= limpio["charuco_cuadro_mm"]:
        avisos.append(
            f"charuco_marcador_mm ({limpio['charuco_marcador_mm']:g}) debe ser MENOR "
            f"que charuco_cuadro_mm ({limpio['charuco_cuadro_mm']:g}): el marcador va "
            f"dentro del cuadro. Usando los valores por defecto de ambos")
        limpio["charuco_cuadro_mm"] = DEFAULTS_CONFIG["charuco_cuadro_mm"]
        limpio["charuco_marcador_mm"] = DEFAULTS_CONFIG["charuco_marcador_mm"]

    return limpio, avisos


def cargar_config():
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
    alto_img, ancho_img = forma_img[:2]
    p = esquinas.reshape(-1, 2)
    return max(float(np.ptp(p[:, 0])) / ancho_img,
               float(np.ptp(p[:, 1])) / alto_img)


def detectar_tablero(gris, tablero=CHECKERBOARD):
    ok, esquinas = cv2.findChessboardCornersSB(gris, tablero, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        return True, esquinas

    ok, esquinas = cv2.findChessboardCorners(gris, tablero, FLAGS_TABLERO)
    if not ok:
        return False, None
    return True, cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), CRITERIA_SUBPIX)


def ordenar_esquinas(esquinas, tablero=CHECKERBOARD):
    cols, filas = tablero
    rejilla = esquinas.reshape(filas, cols, 2)

    mejor, mejor_score = rejilla, None
    for giro in range(4):
        cand = np.rot90(rejilla, giro)
        if cand.shape[:2] != (filas, cols):
            continue
        score = float(cand[0, 0, 0] + cand[0, 0, 1])
        if mejor_score is None or score < mejor_score:
            mejor, mejor_score = cand, score

    return np.ascontiguousarray(mejor.reshape(-1, 1, 2))


def crear_charuco(cfg=None):
    c = cfg or CONFIG
    diccionario = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, c["charuco_diccionario"]))
    board = cv2.aruco.CharucoBoard(
        tuple(c["charuco_cuadros"]),
        float(c["charuco_cuadro_mm"]),
        float(c["charuco_marcador_mm"]),
        diccionario)
    return board, cv2.aruco.CharucoDetector(board)


def detectar_charuco(gris, detector):
    esquinas, ids, _, _ = detector.detectBoard(gris)
    if esquinas is None or ids is None or len(ids) < 4:
        return False, None, None
    return True, esquinas, ids.ravel()


def puntos_de_charuco(board, ids):
    return board.getChessboardCorners()[ids].astype(np.float32)


RANGO_FOV_CREIBLE = (40.0, 130.0)


def fov_grados(fx, ancho_img):
    return 2 * np.degrees(np.arctan(ancho_img / (2 * fx)))


def pares_de_capturas(carpeta=CARPETA_CAPTURAS):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.png')]
    cam0 = {f for f in archivos if f.startswith('cam0_')}
    cam1 = {f for f in archivos if f.startswith('cam1_')}
    indices = sorted({int(f.split('_')[1].split('.')[0]) for f in archivos if '_' in f})
    pares = []
    for i in indices:
        a = f'cam0_{i}.png'
        b = f'cam1_{i}.png'
        if a in archivos and b in archivos:
            pares.append((os.path.join(carpeta, a), os.path.join(carpeta, b)))
    return pares


def buscar_archivo_desesperadamente(nombre):
    """Busca `nombre` en varias rutas razonables del proyecto y devuelve la ruta o None.

    Implementación simple y robusta usada por varios scripts para localizar
    `stereo.npz`, `avatar_config.json`, etc.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lugares = [
        os.path.join(base_dir, "..", "calibracion"),
        os.path.join(base_dir, "..", "data", "calibracion"),
        os.path.join(base_dir, "..", "data"),
        os.path.join(base_dir, ".."),
        base_dir,
        os.path.join(base_dir, "..", ".."),
        os.getcwd(),
    ]
    for ruta in lugares:
        path = os.path.join(ruta, nombre)
        if os.path.exists(path):
            return os.path.abspath(path)
    for root, dirs, files in os.walk(base_dir):
        if nombre in files:
            return os.path.join(root, nombre)
    return None
