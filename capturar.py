import cv2
import os
import time

# === CONFIGURACIÓN ===
# Buscamos la ruta donde está ESTE archivo .py
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))
# Creamos la carpeta de fotos AQUÍ MISMO
CARPETA_GUARDADO = os.path.join(CARPETA_BASE, "capturas")

CHECKERBOARD = (9, 6) 
INTERVALO_SEGUNDOS = 2.0 
TOTAL_FOTOS = 30         

if not os.path.exists(CARPETA_GUARDADO): 
    os.makedirs(CARPETA_GUARDADO)
    print(f"✅ Carpeta de fotos creada en: {CARPETA_GUARDADO}")

cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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

    # Oscurecer un poco la cámara izquierda (como en tu código original)
    frame0 = cv2.convertScaleAbs(frame0, alpha=0.6, beta=-30)

    show0 = frame0.copy()
    show1 = frame1.copy()

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    found0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, None)
    found1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)

    if found0: cv2.drawChessboardCorners(show0, CHECKERBOARD, corners0, found0)
    if found1: cv2.drawChessboardCorners(show1, CHECKERBOARD, corners1, found1)

    if found0 and found1:
        tiempo_actual = time.time()
        
        if (tiempo_actual - ultimo_tiempo) > INTERVALO_SEGUNDOS:
            img_name0 = os.path.join(CARPETA_GUARDADO, f"cam0_{contador}.png")
            img_name1 = os.path.join(CARPETA_GUARDADO, f"cam1_{contador}.png")
            
            cv2.imwrite(img_name0, frame0)
            cv2.imwrite(img_name1, frame1)
            
            print(f"📸 Foto {contador+1} guardada.")
            contador += 1
            ultimo_tiempo = tiempo_actual
            
            cv2.putText(show0, "GUARDADA!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            restante = int(INTERVALO_SEGUNDOS - (tiempo_actual - ultimo_tiempo))
            cv2.putText(show0, str(restante), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Camara 0", cv2.resize(show0, (640, 480)))
    cv2.imshow("Camara 1", cv2.resize(show1, (640, 480)))

    if cv2.waitKey(1) == 27: break

cap0.release(); cap1.release()
cv2.destroyAllWindows()