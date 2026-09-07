import cv2
import numpy as np
import os
import time

from Herramientas import BACKEND_CAMARA, CHECKERBOARD, FLAGS_TABLERO

# === CONFIGURACIÓN ===
# Buscamos la ruta donde está ESTE archivo .py
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
# Creamos la carpeta de fotos AQUÍ MISMO
CARPETA_GUARDADO = os.path.join(CARPETA_BASE, "capturas")

INTERVALO_SEGUNDOS = 2.0
TOTAL_FOTOS = 30

# Fraccion minima del ancho del cuadro que debe ocupar el tablero.
# Es un piso grueso, no una garantia: capturas al 9% del ancho dieron una
# calibracion inservible (focal de 1515 px, o sea 24 grados de campo de vision,
# implausible para una webcam). Lo que de verdad condiciona bien el sistema es
# VARIAR LA INCLINACION del tablero, no su tamano en el cuadro; el tamano solo
# ayuda a localizar mejor las esquinas.
# La verificacion real es a posteriori: que el fx resultante corresponda a un
# campo de vision creible para la camara.
MIN_ANCHO_TABLERO = 0.15


def ancho_relativo(esquinas, ancho_img):
    """Ancho del tablero detectado como fraccion del ancho de la imagen."""
    p = esquinas.reshape(-1, 2)
    return float(np.ptp(p[:, 0])) / ancho_img

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

    found0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, FLAGS_TABLERO)
    found1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, FLAGS_TABLERO)

    if found0: cv2.drawChessboardCorners(show0, CHECKERBOARD, corners0, found0)
    if found1: cv2.drawChessboardCorners(show1, CHECKERBOARD, corners1, found1)

    # Que tan grande se ve el tablero en cada camara
    ancho0 = ancho_relativo(corners0, frame0.shape[1]) if found0 else 0.0
    ancho1 = ancho_relativo(corners1, frame1.shape[1]) if found1 else 0.0
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

    # Una sola ventana con las dos camaras. Antes se abrian dos ventanas del
    # mismo tamano en la misma posicion, asi que una tapaba a la otra.
    ALTO_BARRA = 76
    vistas = []
    for etiqueta, vista, detecto, anc in (("cam0", show0, found0, ancho0),
                                          ("cam1", show1, found1, ancho1)):
        v = cv2.resize(vista, (560, 420))
        cv2.rectangle(v, (0, 0), (560, 26), (0, 0, 0), -1)
        if not detecto:
            txt, col = f"{etiqueta}: NO ve el tablero", (0, 0, 255)
        elif anc < MIN_ANCHO_TABLERO:
            txt, col = (f"{etiqueta}: {anc*100:.0f}% - ACERCALO "
                        f"(min {MIN_ANCHO_TABLERO*100:.0f}%)"), (0, 165, 255)
        else:
            txt, col = f"{etiqueta}: {anc*100:.0f}% del ancho - bien", (0, 255, 0)
        cv2.putText(v, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)
        vistas.append(v)

    panel = cv2.hconcat(vistas)
    barra = np.zeros((ALTO_BARRA, panel.shape[1], 3), np.uint8)

    # Progreso: contador, faltantes y barra de avance
    cv2.putText(barra, f"{contador} de {TOTAL_FOTOS} guardadas   -   faltan {TOTAL_FOTOS - contador}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    x0, x1, y = 10, panel.shape[1] - 10, 40
    cv2.rectangle(barra, (x0, y), (x1, y + 12), (70, 70, 70), -1)
    if contador:
        ancho = int((x1 - x0) * contador / TOTAL_FOTOS)
        cv2.rectangle(barra, (x0, y), (x0 + ancho, y + 12), (0, 200, 0), -1)

    # Estado grande: guardando, cuenta regresiva, o que falta
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