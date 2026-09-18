import sys
from collections import Counter

import cv2
import numpy as np

from utils.Herramientas import BACKEND_CAMARA

NOMBRES = sorted({n.upper() for n in dir(cv2.aruco) if n.startswith("DICT_")})

CUADROS_A_VOTAR = 60
MIN_MARCADORES = 4


def probar_todos(gris):
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
    max_marcadores = Counter()
    max_id = Counter()
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


def reportar_json(votos, max_marcadores, max_id, analizados):
    """Devuelve un dict con el resumen (útil si se llama desde otro programa)."""
    if not votos:
        return {"dictionary": None, "analizados": analizados}
    ganador, n = votos.most_common(1)[0]
    return {
        "dictionary": ganador,
        "detected_frames": int(n),
        "analizados": int(analizados),
        "max_markers": int(max_marcadores[ganador]),
        "max_id": int(max_id[ganador]),
    }


def reportar(votos, max_marcadores, max_id, analizados):
    print()
    if not votos:
        print("Ningun diccionario detecto marcadores.")
        return

    ganador, n = votos.most_common(1)[0]
    marcadores = max_marcadores[ganador]
    mayor_id = max_id[ganador]

    print(f"DICCIONARIO: {ganador}")
    print(f"  detectado en {n} de {analizados} cuadros analizados")
    print(f"  maximo de marcadores vistos a la vez: {marcadores}")
    print(f"  ID mas alto visto: {mayor_id}")

    total = mayor_id + 1
    print(f"\n  Marcadores en el tablero completo: {total} ")


if __name__ == "__main__":
    if "--check" in sys.argv:
        pass
    else:
        indice = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
        main(indice)
