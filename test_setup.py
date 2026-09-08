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


def test_tamano_del_tablero_sobrevive_la_inclinacion():
    """Un tablero inclinado esta igual de cerca, y debe medirse igual de cerca.

    capturar.py pide inclinar el tablero (es lo que condiciona la calibracion)
    pero medía el tamano solo como span horizontal, que la inclinacion sobre el
    eje vertical comprime. Con eso el tablero se detectaba perfecto y la barra
    seguia diciendo ACERCALO, sin forma de satisfacer las dos cosas.
    """
    import numpy as np
    from Herramientas import MIN_ANCHO_TABLERO, tamano_relativo

    FORMA = (480, 640, 3)  # alto, ancho, canales: como frame.shape
    alto, ancho = FORMA[:2]

    def rejilla(esc_x, esc_y):
        """Tablero 7x7 centrado, escalado en cada eje (esc=1.0 llena el cuadro)."""
        xs = np.linspace(-0.5, 0.5, 7) * ancho * esc_x + ancho / 2
        ys = np.linspace(-0.5, 0.5, 7) * alto * esc_y + alto / 2
        return np.array([[[x, y]] for y in ys for x in xs], np.float32)

    # De frente ocupando 40% del cuadro: pasa el minimo con holgura.
    de_frente = tamano_relativo(rejilla(0.40, 0.40), FORMA)
    assert de_frente > MIN_ANCHO_TABLERO, f"de frente al 40% dio {de_frente:.3f}"

    # Inclinado 70 grados sobre el eje vertical: el span horizontal cae a
    # cos(70)=34% del original (13% del cuadro, bajo el minimo) pero el
    # vertical no cambia. El tablero no se alejo, asi que debe seguir pasando.
    inclinado = tamano_relativo(rejilla(0.40 * np.cos(np.radians(70)), 0.40), FORMA)
    assert inclinado > MIN_ANCHO_TABLERO, \
        f"inclinar 70 grados tumba la medida a {inclinado:.3f}; volvio el bug del span en X"

    # Lo mismo inclinando sobre el eje horizontal: ahora sobrevive el span en X.
    assert tamano_relativo(rejilla(0.40, 0.40 * np.cos(np.radians(70))), FORMA) > MIN_ANCHO_TABLERO

    # Y un tablero de verdad lejos sigue reprobando: el piso no se volvio inutil.
    lejos = tamano_relativo(rejilla(0.09, 0.09), FORMA)
    assert lejos < MIN_ANCHO_TABLERO, f"al 9% del cuadro deberia reprobar, dio {lejos:.3f}"


def test_detectar_tablero_encuentra_las_esquinas_internas():
    """detectar_tablero devuelve las cols*filas esquinas internas, ya en subpixel."""
    import numpy as np
    from Herramientas import CHECKERBOARD, detectar_tablero

    cols, filas = CHECKERBOARD
    lado = 40
    # (cols+1) x (filas+1) cuadros producen cols x filas esquinas internas
    patron = np.indices((filas + 1, cols + 1)).sum(axis=0) % 2
    tablero = (np.kron(patron, np.ones((lado, lado))) * 255).astype(np.uint8)
    tablero = cv2.copyMakeBorder(tablero, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

    ok, esquinas = detectar_tablero(tablero)
    assert ok, "no detecta un tablero sintetico limpio"
    p = esquinas.reshape(-1, 2)
    assert p.shape[0] == cols * filas, f"esperadas {cols * filas} esquinas, dio {p.shape[0]}"
    # Subpixel: la primera esquina interna cae en el borde del primer cuadro.
    assert np.allclose(p.min(axis=0), [40 + lado, 40 + lado], atol=1.0), \
        f"esquinas mal localizadas: min en {p.min(axis=0)}"


def test_config_rechaza_valores_invalidos():
    """config.json es una frontera de confianza: lo edita un menu y una persona.

    Un cero en checkerboard o un tamano de cuadro negativo produciria una
    calibracion basura sin ningun error visible, asi que un valor fuera de
    rango tiene que caer al default y avisar, nunca pasar de largo.
    """
    from Herramientas import DEFAULTS_CONFIG, validar_config

    limpio, avisos = validar_config({})
    assert limpio == DEFAULTS_CONFIG and not avisos, "un dict vacio son los defaults"

    limpio, avisos = validar_config({"checkerboard": [9, 6], "tamano_cuadro_mm": 25,
                                     "tablero_plegado": True})
    assert limpio["checkerboard"] == [9, 6]
    assert limpio["tamano_cuadro_mm"] == 25.0
    assert limpio["tablero_plegado"] is True
    assert not avisos, f"valores validos no deben avisar: {avisos}"

    malos = [
        {"checkerboard": [0, 7]},            # una rejilla de 0 no existe
        {"checkerboard": [7]},               # falta una dimension
        {"checkerboard": "7x7"},             # no es lista
        {"checkerboard": [7.5, 7]},          # no entero
        {"tamano_cuadro_mm": -20},           # negativo: escala invertida
        {"tamano_cuadro_mm": 0},             # cero: escala colapsada
        {"tamano_cuadro_mm": "20"},          # texto
        {"min_ancho_tablero": 1.5},          # mas del 100% del cuadro
        {"total_fotos": 2},                  # menos del minimo para calibrar
        {"total_fotos": True},               # bool no es un numero valido
        {"intervalo_segundos": 0},           # 30 fotos identicas
        {"tablero_plegado": "si"},           # texto en una clave si/no
        {"tablero_plegado": 1},              # un 1 no es true
    ]
    for crudo in malos:
        clave = next(iter(crudo))
        limpio, avisos = validar_config(crudo)
        assert limpio[clave] == DEFAULTS_CONFIG[clave], \
            f"{crudo} se acepto y deberia caer al default"
        assert avisos, f"{crudo} se rechazo en silencio, sin aviso"


def test_guardar_config_no_pierde_ediciones_previas():
    """Editar el tablero y luego el cuadro debe conservar los dos.

    La primera version mezclaba sobre la CONFIG del import, que nunca cambia,
    asi que el segundo guardado revertia el primero.
    """
    import json
    import tempfile
    import Herramientas as H

    original = H.ARCHIVO_CONFIG
    with tempfile.TemporaryDirectory() as tmp:
        H.ARCHIVO_CONFIG = os.path.join(tmp, "config.json")
        try:
            H.guardar_config({"checkerboard": [9, 6]})
            guardada, _ = H.guardar_config({"tamano_cuadro_mm": 25.0})

            assert guardada["checkerboard"] == [9, 6], \
                f"se perdio el tablero al guardar el cuadro: {guardada}"
            assert guardada["tamano_cuadro_mm"] == 25.0

            with open(H.ARCHIVO_CONFIG, encoding="utf-8") as f:
                del_disco = json.load(f)
            assert del_disco["checkerboard"] == [9, 6], "el disco no coincide"

            # Un valor invalido no se escribe: se guarda el default y avisa.
            guardada, avisos = H.guardar_config({"tamano_cuadro_mm": -5})
            assert avisos and guardada["tamano_cuadro_mm"] == 20.0
        finally:
            H.ARCHIVO_CONFIG = original


def test_pares_de_capturas_solo_empareja_indices_completos():
    """El menu cuenta pares con esta funcion y generar_calibracion.py calibra
    con ella: si contaran distinto, el menu diria 18 y el calibrador 12."""
    import tempfile
    from Herramientas import pares_de_capturas

    with tempfile.TemporaryDirectory() as tmp:
        for nombre in ("cam0_0.png", "cam1_0.png",      # par bueno
                       "cam0_1.png", "cam1_1.png",      # par bueno
                       "cam0_2.png",                    # sin su cam1
                       "cam1_3.png",                    # sin su cam0
                       "cam0_4.jpg", "cam1_4.png",      # extension distinta
                       "cam0_x.png", "cam1_x.png",      # indice no numerico
                       "notas.txt"):
            open(os.path.join(tmp, nombre), "w").close()

        pares = pares_de_capturas(tmp)
        assert len(pares) == 2, f"esperados 2 pares, dio {len(pares)}: {pares}"
        assert [os.path.basename(a) for a, _ in pares] == ["cam0_0.png", "cam0_1.png"]
        assert [os.path.basename(b) for _, b in pares] == ["cam1_0.png", "cam1_1.png"]

    # Una carpeta que no existe son cero pares, no una excepcion: el menu la
    # consulta antes de que capturar.py se haya corrido nunca.
    assert pares_de_capturas(os.path.join(tempfile.gettempdir(), "no_existe_xyz")) == []


def test_fov_delata_la_calibracion_mal_condicionada():
    """El chequeo que detecta una calibracion mala aunque su RMS se vea bien."""
    from Herramientas import RANGO_FOV_CREIBLE, fov_grados

    bajo, alto = RANGO_FOV_CREIBLE

    # El caso real de este proyecto: 1515 px de focal en un cuadro de 640.
    malo = fov_grados(1515, 640)
    assert abs(malo - 23.9) < 0.1, f"esperado ~23.9 grados, dio {malo:.1f}"
    assert not bajo < malo < alto, "24 grados no es creible para una webcam"

    # Una webcam normal: 530 px en 640 son ~62 grados.
    bueno = fov_grados(530.6, 640)
    assert bajo < bueno < alto, f"{bueno:.1f} grados deberia ser creible"


if __name__ == "__main__":
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("OK", nombre)
