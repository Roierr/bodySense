"""Averigua que tablero CharuCo tienes, apuntandole con la camara.

Un CharuCo se define con cuatro datos y los cuatro deben coincidir EXACTAMENTE
con el tablero impreso. Tres se sacan mirando: cuantos cuadros de ancho, de
alto, y los dos tamanos en milimetros con una regla. El cuarto, el
diccionario, no se puede adivinar: hay 27 y el tablero no dice cual usa.

Este programa lo averigua probando los 27 contra lo que ve la camara. Con el
diccionario correcto detecta los marcadores; con cualquier otro detecta cero.
Es una senal muy limpia, no hay medias tintas.

No necesita saber nada del tablero para funcionar, y no toca ningun archivo:
solo mira y reporta.

    python3 medir_charuco.py           usa la camara 0
    python3 medir_charuco.py 1         usa la camara 1

Apunta la camara al tablero completo, con buena luz, y espera unos segundos.
ESC para salir antes de tiempo.
"""
import sys
from collections import Counter

import cv2
import numpy as np

from Herramientas import BACKEND_CAMARA

# Los 27 diccionarios de esta version de OpenCV, sin repetidos: varios existen
# con el mismo nombre en mayuscula y minuscula (DICT_APRILTAG_36H11 y 36h11).
NOMBRES = sorted({n.upper() for n in dir(cv2.aruco) if n.startswith("DICT_")})

CUADROS_A_VOTAR = 60      # cuantos cuadros de video analizar antes de concluir
MIN_MARCADORES = 4        # menos de esto es ruido, no un tablero


def probar_todos(gris):
    """Devuelve {nombre_diccionario: cuantos marcadores detecto} para los que detectaron."""
    hallazgos = {}
    for nombre in NOMBRES:
        valor = getattr(cv2.aruco, nombre, None)
        if valor is None:
            continue
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(valor),
            cv2.aruco.DetectorParameters())
        _, ids, _ = detector.detectMarkers(gris)
        if ids is not None and len(ids) >= MIN_MARCADORES:
            hallazgos[nombre] = (len(ids), int(ids.max()))
    return hallazgos


def main(indice=0):
    cap = cv2.VideoCapture(indice, BACKEND_CAMARA)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: no abre la camara {indice}.")

    print(f"Probando {len(NOMBRES)} diccionarios contra la camara {indice}.")
    print("Apunta al tablero completo, con buena luz. ESC para salir.\n")

    votos = Counter()
    max_marcadores = Counter()   # por diccionario, el maximo visto de una vez
    max_id = Counter()           # por diccionario, el ID mas alto visto
    analizados = 0

    while analizados < CUADROS_A_VOTAR:
        ok, frame = cap.read()
        if not ok:
            continue
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hallazgos = probar_todos(gris)
        analizados += 1

        for nombre, (cuantos, mayor) in hallazgos.items():
            votos[nombre] += 1
            max_marcadores[nombre] = max(max_marcadores[nombre], cuantos)
            max_id[nombre] = max(max_id[nombre], mayor)

        vista = cv2.resize(frame, (640, 480))
        cv2.rectangle(vista, (0, 0), (640, 52), (0, 0, 0), -1)
        if votos:
            mejor, n = votos.most_common(1)[0]
            texto = f"{mejor}  ({max_marcadores[mejor]} marcadores)"
            color = (0, 255, 0)
        else:
            texto = "ningun diccionario responde todavia"
            color = (0, 165, 255)
        cv2.putText(vista, texto, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vista, f"analizando {analizados}/{CUADROS_A_VOTAR}",
                    (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Identificando tablero CharuCo - ESC para salir", vista)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    reportar(votos, max_marcadores, max_id, analizados)


def reportar(votos, max_marcadores, max_id, analizados):
    print()
    if not votos:
        print("Ningun diccionario detecto marcadores.")
        print("Revisa:")
        print("  1. Que el tablero sea CharuCo y no de ajedrez normal")
        print("     (un CharuCo tiene un codigo impreso dentro de cada cuadro blanco)")
        print("  2. Que haya luz suficiente y el tablero se vea completo y enfocado")
        print("  3. Que la camara sea la correcta: prueba 'python3 medir_charuco.py 1'")
        return

    ganador, n = votos.most_common(1)[0]
    marcadores = max_marcadores[ganador]
    mayor_id = max_id[ganador]

    print(f"DICCIONARIO: {ganador}")
    print(f"  detectado en {n} de {analizados} cuadros analizados")
    print(f"  maximo de marcadores vistos a la vez: {marcadores}")
    print(f"  ID mas alto visto: {mayor_id}")

    if len(votos) > 1:
        print("\n  Otros que tambien respondieron (normal: varios diccionarios comparten codigos):")
        for nombre, v in votos.most_common()[1:4]:
            print(f"    {nombre}: {v} cuadros, hasta {max_marcadores[nombre]} marcadores")
        print("  Quedate con el de mas votos y mas marcadores.")

    # Un CharuCo de C x F cuadros lleva la mitad de marcadores, redondeando.
    # Del total de marcadores se puede estimar cuantos cuadros tiene.
    total = mayor_id + 1
    print(f"\n  Marcadores en el tablero completo: {total} "
          f"(si viste el tablero entero)")
    print(f"  Eso corresponde a un tablero de ~{total * 2} cuadros, por ejemplo "
          f"{sugerir(total * 2)}")

    print("\n" + "=" * 64)
    print("Ponlo en config.json asi (falta que midas los dos tamanos con regla):")
    print("=" * 64)
    print(f"""{{
  "tablero_tipo": "charuco",
  "charuco_diccionario": "{ganador}",
  "charuco_cuadros": [COLUMNAS, FILAS],
  "charuco_cuadro_mm": LADO_DEL_CUADRO,
  "charuco_marcador_mm": LADO_DEL_CODIGO_NEGRO
}}""")
    print("\nCOLUMNAS y FILAS: cuentalos a ojo en el tablero (cuadros, no esquinas).")
    print("Los dos tamanos: con regla, en milimetros. El del codigo va DENTRO")
    print("del cuadro blanco, asi que siempre es menor que el del cuadro.")


def sugerir(cuadros):
    """Pares de factores plausibles para ese total de cuadros."""
    opciones = [f"{c}x{cuadros // c}" for c in range(3, int(cuadros ** 0.5) + 1)
                if cuadros % c == 0]
    return ", ".join(opciones) if opciones else "no se pudo factorizar"


def demo():
    """Chequeo sin camara: el identificador reconoce un tablero que generamos."""
    esperado = "DICT_5X5_100"
    diccionario = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, esperado))
    board = cv2.aruco.CharucoBoard((5, 7), 30.0, 22.0, diccionario)
    imagen = board.generateImage((600, 840), marginSize=20)

    hallazgos = probar_todos(imagen)
    assert esperado in hallazgos, \
        f"no identifico {esperado}; encontro {list(hallazgos)}"
    cuantos, _ = hallazgos[esperado]
    assert cuantos >= 10, f"solo detecto {cuantos} marcadores de 17"

    # Un tablero de ajedrez normal no debe disparar ningun diccionario.
    patron = (np.indices((8, 8)).sum(axis=0) % 2)
    ajedrez = (np.kron(patron, np.ones((60, 60))) * 255).astype(np.uint8)
    assert not probar_todos(ajedrez), \
        "un tablero de ajedrez normal no deberia detectar marcadores ArUco"

    print(f"OK: identifico {esperado} con {cuantos} marcadores, "
          f"y descarta un tablero de ajedrez normal")


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo()
    else:
        indice = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
        main(indice)
