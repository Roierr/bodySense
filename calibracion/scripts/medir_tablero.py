import sys
from collections import Counter

import cv2

from utils.Herramientas import BACKEND_CAMARA, FLAGS_TABLERO

CANDIDATAS = [(c, r) for c in range(11, 4, -1) for r in range(c, 3, -1)]
CANDIDATAS.sort(key=lambda cr: -(cr[0] * cr[1]))


def medir(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for cols, rows in CANDIDATAS:
        ok, esquinas = cv2.findChessboardCorners(gris, (cols, rows), FLAGS_TABLERO)
        if ok:
            return (cols, rows), esquinas
    return None, None


def main():
    indice = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    cap = cv2.VideoCapture(indice, BACKEND_CAMARA)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: no abre la camara {indice}.")

    print(f"Camara {indice}. Sosten el tablero COMPLETO frente a ella.")
    print("Probando rejillas... (ESC para salir)\n")

    votos = Counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        rejilla, esquinas = medir(frame)
        vista = frame.copy()

        if rejilla:
            votos[rejilla] += 1
            cv2.drawChessboardCorners(vista, rejilla, esquinas, True)
            cols, rows = rejilla
            txt = f"({cols}, {rows}) esquinas  =  tablero de {cols+1} x {rows+1} cuadros"
            colorTxt = (0, 255, 0)
        else:
            txt = "no detecto tablero"
            colorTxt = (0, 0, 255)

        cv2.rectangle(vista, (0, 0), (vista.shape[1], 58), (0, 0, 0), -1)
        cv2.putText(vista, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorTxt, 1)
        if votos:
            mejor, n = votos.most_common(1)[0]
            cv2.putText(vista, f"mas estable: {mejor}  ({n} cuadros)", (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(f"Medir tablero - camara {indice}", vista)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not votos:
        print("No se detecto ningun tablero. Revisa luz, enfoque, y que quepa completo.")
        return
    print("Resultados (rejilla de esquinas internas -> cuadros detectados):")
    for (cols, rows), n in votos.most_common():
        print(f"  ({cols}, {rows})  tablero {cols+1} x {rows+1} cuadros   {n} cuadros de video")
    mejor = votos.most_common(1)[0][0]
    print(f"\nPon esto en capturar.py y generar_calibracion.py:")
    print(f"  CHECKERBOARD = {mejor}")
    print("Y mide un cuadro con regla para TAMANO_CUADRO (en mm).")


if __name__ == "__main__":
    if "--check" in sys.argv:
        pass
    else:
        main()
