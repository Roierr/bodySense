import sys
import cv2
import numpy as np
import os
import time

from utils.Herramientas import (BACKEND_CAMARA, CARPETA_CAPTURAS, CHECKERBOARD,
                          INTERVALO_SEGUNDOS, MIN_ANCHO_TABLERO, TOTAL_FOTOS,
                          detectar_tablero, tamano_relativo, crear_charuco,
                          detectar_charuco, CONFIG)

MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "ajedrez"

CARPETA_GUARDADO = CARPETA_CAPTURAS

# Preparar detector CharuCo si se pidio
if MODE == "charuco":
    board, charuco_detector = crear_charuco()
else:
    charuco_detector = None

# Parámetros para estimar distancia aproximada (solo orientación)
ASSUMED_FOV_DEG = 70.0


if not os.path.exists(CARPETA_GUARDADO):
    os.makedirs(CARPETA_GUARDADO)
    print(f"✅ Carpeta de fotos creada en: {CARPETA_GUARDADO}")

cap0 = cv2.VideoCapture(0, BACKEND_CAMARA)
cap1 = cv2.VideoCapture(1, BACKEND_CAMARA)

if not cap0.isOpened() or not cap1.isOpened():
    print("❌ Error: No se detectan las cámaras.")
    exit()

print(f"--- MODO CAPTURA ---")
print(f"Guardando en: {CARPETA_GUARDADO}")

contador = 0
ultimo_tiempo = 0

while contador < TOTAL_FOTOS:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    
    if not ret0 or not ret1: continue

    show0 = frame0.copy()
    show1 = frame1.copy()

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    if MODE == "charuco":
        found0, corners0, ids0 = detectar_charuco(gray0, charuco_detector)
        found1, corners1, ids1 = detectar_charuco(gray1, charuco_detector)

        # dibujar esquinas CharuCo simples
        if found0:
            for c in corners0.reshape(-1, 2):
                cv2.circle(show0, tuple(map(int, c)), 4, (0, 255, 0), -1)
        if found1:
            for c in corners1.reshape(-1, 2):
                cv2.circle(show1, tuple(map(int, c)), 4, (0, 255, 0), -1)

        ancho0 = tamano_relativo(corners0, frame0.shape) if found0 else 0.0
        ancho1 = tamano_relativo(corners1, frame1.shape) if found1 else 0.0

        # estimacion aproximada de distancia (mm) usando tamaño conocido del charuco
        real_square_mm = float(CONFIG.get("charuco_cuadro_mm", 30.0))
        cols, rows = CONFIG.get("charuco_cuadros", (5, 7))
        real_width_mm = (cols - 1) * real_square_mm
        def _estimate_dist_mm(corners, forma):
            p = corners.reshape(-1, 2)
            observed_px = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])))
            if observed_px <= 0:
                return None
            ancho_img = forma[1]
            fx_est = ancho_img / (2.0 * np.tan(np.deg2rad(ASSUMED_FOV_DEG / 2.0)))
            return (real_width_mm * fx_est) / observed_px

        dist0 = _estimate_dist_mm(corners0, frame0.shape) if found0 else None
        dist1 = _estimate_dist_mm(corners1, frame1.shape) if found1 else None
    else:
        found0, corners0 = detectar_tablero(gray0)
        found1, corners1 = detectar_tablero(gray1)

        if found0:
            cv2.drawChessboardCorners(show0, CHECKERBOARD, corners0, found0)
        if found1:
            cv2.drawChessboardCorners(show1, CHECKERBOARD, corners1, found1)

        ancho0 = tamano_relativo(corners0, frame0.shape) if found0 else 0.0
        ancho1 = tamano_relativo(corners1, frame1.shape) if found1 else 0.0

        # estimacion aproximada de distancia para checkerboard
        real_square_mm = float(CONFIG.get("tamano_cuadro_mm", 20.0))
        cols, rows = CHECKERBOARD
        real_width_mm = (cols - 1) * real_square_mm
        def _estimate_dist_mm_cb(corners, forma):
            p = corners.reshape(-1, 2)
            observed_px = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])))
            if observed_px <= 0:
                return None
            ancho_img = forma[1]
            fx_est = ancho_img / (2.0 * np.tan(np.deg2rad(ASSUMED_FOV_DEG / 2.0)))
            return (real_width_mm * fx_est) / observed_px

        dist0 = _estimate_dist_mm_cb(corners0, frame0.shape) if found0 else None
        dist1 = _estimate_dist_mm_cb(corners1, frame1.shape) if found1 else None
    cerca = ancho0 >= MIN_ANCHO_TABLERO and ancho1 >= MIN_ANCHO_TABLERO

    guardada = False
    espera = None
    if found0 and found1 and cerca:
        tiempo_actual = time.time()

        if (tiempo_actual - ultimo_tiempo) > INTERVALO_SEGUNDOS:
            img_name0 = os.path.join(CARPETA_GUARDADO, f"cam0_{contador}.png")
            img_name1 = os.path.join(CARPETA_GUARDADO, f"cam1_{contador}.png")

            cv2.imwrite(img_name0, frame0)
            cv2.imwrite(img_name1, frame1)

            contador += 1
            ultimo_tiempo = tiempo_actual
            guardada = True
            print(f"📸 Foto {contador} de {TOTAL_FOTOS} guardada. Faltan {TOTAL_FOTOS - contador}.")
        else:
            espera = INTERVALO_SEGUNDOS - (tiempo_actual - ultimo_tiempo)

    ALTO_BARRA = 76
    vistas = []
    for etiqueta, vista, detecto, anc in (("cam0", show0, found0, ancho0),
                                          ("cam1", show1, found1, ancho1)):
        v = cv2.resize(vista, (560, 420))
        cv2.rectangle(v, (0, 0), (560, 26), (0, 0, 0), -1)
        # Mensajes mas verbosos: porcentaje y estimacion de distancia si hay
        if not detecto:
            txt, col = f"{etiqueta}: NO ve el tablero", (0, 0, 255)
        else:
            # encontrar la distancia estimada correspondiente
            d = dist0 if etiqueta == "cam0" else dist1
            dist_text = f" - ~{d/10:.1f} cm" if (d is not None and d > 0) else ""
            if anc < MIN_ANCHO_TABLERO:
                txt, col = (f"{etiqueta}: {anc*100:.0f}% - ACERCALO {dist_text} "
                            f"(min {MIN_ANCHO_TABLERO*100:.0f}%)"), (0, 165, 255)
            elif anc > 0.6:
                txt, col = (f"{etiqueta}: {anc*100:.0f}% - ALEJA {dist_text}"), (0, 165, 255)
            else:
                txt, col = f"{etiqueta}: {anc*100:.0f}% del cuadro{dist_text} - bien", (0, 255, 0)
        cv2.putText(v, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)
        vistas.append(v)

    panel = cv2.hconcat(vistas)
    barra = np.zeros((ALTO_BARRA, panel.shape[1], 3), np.uint8)

    cv2.putText(barra, f"{contador} de {TOTAL_FOTOS} guardadas   -   faltan {TOTAL_FOTOS - contador}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    x0, x1, y = 10, panel.shape[1] - 10, 40
    cv2.rectangle(barra, (x0, y), (x1, y + 12), (70, 70, 70), -1)
    if contador:
        ancho = int((x1 - x0) * contador / TOTAL_FOTOS)
        cv2.rectangle(barra, (x0, y), (x0 + ancho, y + 12), (0, 200, 0), -1)

    if guardada:
        msg, col = "GUARDADA!", (0, 255, 0)
    elif espera is not None:
        msg, col = f"siguiente en {espera:.1f} s   -   inclina o mueve el tablero", (0, 255, 255)
    elif found0 and found1 and not cerca:
        msg, col = (f"ACERCA EL TABLERO   cam0 {ancho0*100:.0f}%  cam1 {ancho1*100:.0f}%  "
                    f"(necesitas {MIN_ANCHO_TABLERO*100:.0f}% en las dos)"), (0, 165, 255)
    elif found0 or found1:
        cual = "cam1" if found0 else "cam0"
        msg, col = f"esperando que {cual} vea el tablero completo", (0, 165, 255)
    else:
        msg, col = "ninguna camara ve el tablero", (0, 0, 255)
    cv2.putText(barra, msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)

    cv2.imshow("Captura de calibracion - ESC para salir", cv2.vconcat([barra, panel]))

    if cv2.waitKey(1) == 27: break

cap0.release(); cap1.release()
cv2.destroyAllWindows()

print()
if contador >= TOTAL_FOTOS:
    print(f"Listo: {contador} pares guardados. Sigue: python3 generar_calibracion.py")
elif contador >= 15:
    print(f"Cortaste en {contador} pares. Alcanza para calibrar, pero 30 sale mejor.")
else:
    print(f"Solo {contador} pares. generar_calibracion.py pide al menos 10 y "
          f"recomienda 15+. Vuelve a correr capturar.py.")
