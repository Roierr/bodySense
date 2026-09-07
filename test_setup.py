"""Chequeo minimo: encoding de requirements, backend de camara y rutas de calibracion."""
import os
import sys

import cv2

from Herramientas import BACKEND_CAMARA, CARPETA_CALIBRACION, buscar_archivo_desesperadamente

BASE = os.path.dirname(os.path.abspath(__file__))


def test_archivos_de_texto_son_utf8():
    for nombre in ("requirements.txt", ".gitignore"):
        with open(os.path.join(BASE, nombre), encoding="utf-8") as f:
            assert f.read().strip(), f"{nombre} vacio"


def test_requirements_lista_las_dependencias():
    with open(os.path.join(BASE, "requirements.txt"), encoding="utf-8") as f:
        paquetes = {linea.strip().lower() for linea in f if linea.strip()}
    assert {"opencv-python", "mediapipe", "numpy", "pygame", "pyopengl"} <= paquetes


def test_backend_de_camara_por_plataforma():
    esperado = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    assert BACKEND_CAMARA == esperado
    # DSHOW no existe fuera de Windows: usarlo ahi abre 0 camaras.
    if sys.platform != "win32":
        assert BACKEND_CAMARA != cv2.CAP_DSHOW


def test_encuentra_stereo_npz_sin_walk():
    ruta = buscar_archivo_desesperadamente("stereo.npz")
    assert ruta, "stereo.npz no encontrado; corre generar_calibracion.py"
    assert os.path.dirname(os.path.abspath(ruta)) == CARPETA_CALIBRACION


def test_triangula_bien_con_camaras_de_distinta_resolucion():
    """Dos camaras virtuales de tamano distinto, un punto 3D conocido.

    Escalando los landmarks con las medidas de CADA camara se recupera el punto.
    Escalando ambos con las de la camara 0 (el bug viejo) no.
    """
    import numpy as np

    RES0, RES1 = (1920, 1080), (1280, 720)
    K0 = np.array([[1400.0, 0, RES0[0] / 2], [0, 1400.0, RES0[1] / 2], [0, 0, 1]])
    K1 = np.array([[900.0, 0, RES1[0] / 2], [0, 900.0, RES1[1] / 2], [0, 0, 1]])
    R, T = np.eye(3), np.array([[-400.0], [0.0], [0.0]])  # baseline 400 mm

    P0 = K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K1 @ np.hstack((R, T))

    punto = np.array([50.0, 100.0, 2000.0])

    def proyectar(P, X):
        h = P @ np.append(X, 1.0)
        return h[:2] / h[2]

    px0, px1 = proyectar(P0, punto), proyectar(P1, punto)
    # MediaPipe entrega los landmarks normalizados 0-1 sobre su propio frame.
    norm0 = px0 / RES0
    norm1 = px1 / RES1

    def triangular(a, b):
        p = cv2.triangulatePoints(P0, P1, a.reshape(2, 1), b.reshape(2, 1))
        return (p[:3] / p[3]).ravel()

    bien = triangular(norm0 * RES0, norm1 * RES1)
    assert np.allclose(bien, punto, atol=1e-6), f"esperado {punto}, dio {bien}"

    # Regresion: usar la resolucion de la cam 0 para las dos rompe el resultado.
    mal = triangular(norm0 * RES0, norm1 * RES0)
    assert not np.allclose(mal, punto, atol=1.0), "el bug de resolucion no se detecta"


def test_ordenar_esquinas_resuelve_la_ambiguedad():
    """Un tablero cuadrado numerado desde cuatro esquinas distintas debe
    quedar igual tras reordenar. Si no, la calibracion estereo se cruza."""
    import numpy as np
    from Herramientas import ordenar_esquinas

    C = R = 7
    base = np.array([[[c * 10.0 + 100, r * 10.0 + 50]] for r in range(R) for c in range(C)],
                    np.float32)
    canonica = ordenar_esquinas(base, (C, R))

    for giro in range(1, 4):
        girada = np.ascontiguousarray(
            np.rot90(base.reshape(R, C, 2), giro).reshape(-1, 1, 2))
        assert np.allclose(ordenar_esquinas(girada, (C, R)), canonica), \
            f"el giro de {giro*90} grados no se normaliza"

    # La primera esquina debe ser la de arriba-izquierda
    p = canonica.reshape(-1, 2)
    assert np.allclose(p[0], [100, 50]), f"primera esquina en {p[0]}, esperada [100, 50]"

    # Un tablero rectangular tiene ambiguedad de 180 grados, tambien se normaliza
    C2, R2 = 9, 6
    base2 = np.array([[[c * 10.0, r * 10.0]] for r in range(R2) for c in range(C2)], np.float32)
    vuelta = np.ascontiguousarray(np.rot90(base2.reshape(R2, C2, 2), 2).reshape(-1, 1, 2))
    assert np.allclose(ordenar_esquinas(base2, (C2, R2)),
                       ordenar_esquinas(vuelta, (C2, R2))), "rectangular no se normaliza"


if __name__ == "__main__":
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("OK", nombre)
