import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import json
import sys

# === Importamos los archivos ===
from Herramientas import buscar_archivo_desesperadamente, BACKEND_CAMARA
from Graficos import dibujar_hueso, dibujar_joint, dibujar_cabeza

# ==========================================
# MAIN
# ==========================================
def main():
    stereo = buscar_archivo_desesperadamente("stereo.npz")
    if not stereo: print("\nERROR: Falta stereo.npz"); return
    
    st = np.load(stereo, allow_pickle=True)
    PL = st["mtx1"] @ np.hstack((np.eye(3), np.zeros((3,1))))
    PR = st["mtx2"] @ np.hstack((st["R"], st["T"].reshape(3,1)))


# Color por defecto por si acaso no hay una configuracion previa
    color_final = (0.2, 0.6, 1.0) 
    # Intentar cargar la configuracion del avatar
    cfg = buscar_archivo_desesperadamente("avatar_config.json")
    # Si existe, cargar el color de piel
    if cfg:
        try:
            with open(cfg) as f:
                d = json.load(f)
                h = d["color_piel"].lstrip('#')
                color_final = tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))
        except: pass

        # Inicializar Pygame y OpenGL

    pygame.init()
    pygame.display.set_mode((1000, 800), DOUBLEBUF | OPENGL)

    # La proyeccion se fija una vez, en su propia matriz. Antes se multiplicaba
    # sobre la modelview, que impedia recomponer la vista en cada cuadro.
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (1000/800), 0.1, 50000.0)
    glMatrixMode(GL_MODELVIEW)

    glDisable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    # Vista orbitable. Girar la vista es lo que hace VISIBLE la profundidad:
    # de frente, alejarse solo hace el avatar mas chico (igual que en 2D); de
    # perfil, alejarse lo mueve de lado a lado, que una deteccion 2D no puede
    # producir. El zoom en Y y Z centra el avatar ya escalado en la ventana.
    orbita = 180.0     # grados alrededor del eje vertical
    elevacion = 0.0    # grados sobre el horizonte
    distancia = 900.0  # unidades de la camara virtual al centro del avatar

    #  Establece la confianza mínima (0.1) que un punto corporal detectado por MediaPipe debe tener en ambas cámaras para ser considerado válido y usado en la reconstrucción 3D.
    TOLERANCIA_VISIBILIDAD = 0.1 

# Configurar MediaPipe Pose y las cámaras
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils  # para dibujar los puntos sobre el preview
    pose0 = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4, model_complexity=0)
    pose1 = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4, model_complexity=0)
    
    
    cap0 = cv2.VideoCapture(0, BACKEND_CAMARA); cap1 = cv2.VideoCapture(1, BACKEND_CAMARA)
    if not cap0.isOpened() or not cap1.isOpened():
        print("\nERROR: No se detectan las dos camaras."); pygame.quit(); return
    
    #Esto es para ver el avatar mas grande en la pantalla
    ESCALA_GIGANTE = 1.5

    # Definir las conexiones entre los puntos del cuerpo para dibujar el esqueleto estos puntos son los mismos que usa mediapipe y son universales 
    CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

    # Distancia real del sujeto a las camaras, en mm. Se muestra en pantalla
    # porque es el dato que MediaPipe no puede dar: su 3D esta centrado en las
    # caderas del sujeto y no contiene la posicion absoluta.
    dist_camara = None
    pivote = None      # punto de la sala que la vista orbita, se ancla una vez

    # Bucle principal
    while True:
        # Manejar eventos de Pygame
        for event in pygame.event.get():

            if event.type == QUIT: pygame.quit(); return
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                pygame.quit(); return

        # Girar la vista con las flechas, acercar y alejar con Z y X
        teclas = pygame.key.get_pressed()
        if teclas[K_LEFT]:  orbita -= 1.6
        if teclas[K_RIGHT]: orbita += 1.6
        if teclas[K_UP]:    elevacion = min(85.0, elevacion + 1.2)
        if teclas[K_DOWN]:  elevacion = max(-85.0, elevacion - 1.2)
        if teclas[K_z]:     distancia = max(200.0, distancia - 25.0)
        if teclas[K_x]:     distancia = min(8000.0, distancia + 25.0)
        if teclas[K_r]:
            orbita, elevacion, distancia = 180.0, 0.0, 900.0
            pivote = None   # vuelve a anclar donde este el sujeto ahora

# Leer cuadros de ambas cámaras
        ret0, frame0 = cap0.read(); ret1, frame1 = cap1.read()
        if not ret0 or not ret1: continue

# Procesar con MediaPipe Pose
        res0 = pose0.process(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
        res1 = pose1.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        
        # Limpiar pantalla OpenGL
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

# Reconstruir y dibujar el esqueleto 3D
        if res0.pose_landmarks and res1.pose_landmarks:
            
            lm0, lm1 = res0.pose_landmarks.landmark, res1.pose_landmarks.landmark
            # Obtener puntos visibles en ambas cámaras
            ptsL, ptsR, v_ids = [], [], []
            # Cada camara puede tener resolucion distinta, asi que los landmarks
            # normalizados (0-1) de cada una se escalan con SUS propias medidas.
            h0, w0 = frame0.shape[:2]
            h1, w1 = frame1.shape[:2]

            #Evaluar los 33 puntos de referencia del cuerpo
            for i in range(33):
                # Verificar visibilidad en ambas cámaras, Si ambos puntos son visibles, agregar a la lista
                if lm0[i].visibility > TOLERANCIA_VISIBILIDAD and lm1[i].visibility > TOLERANCIA_VISIBILIDAD:
                    ptsL.append([lm0[i].x * w0, lm0[i].y * h0])
                    ptsR.append([lm1[i].x * w1, lm1[i].y * h1])
                    v_ids.append(i)
            
            # Triangular puntos 3D si hay puntos visibles
            if ptsL:
                p3d = cv2.triangulatePoints(PL, PR, np.array(ptsL).T, np.array(ptsR).T)
                p3d = (p3d[:3] / p3d[3]).T

                # Distancia real a las camaras ANTES de escalar para el render:
                # la Z de la triangulacion esta en milimetros del mundo.
                dist_camara = float(np.median(p3d[:, 2]))

                # Escalar el modelo 3D para una mejor visualización
                p3d = p3d * ESCALA_GIGANTE

# Preparar puntos finales para dibujar
                final_pts = {vid: [p[0], -p[1], p[2]] for vid, p in zip(v_ids, p3d)}

                # El pivote de la vista se fija UNA vez, en el primer cuerpo
                # reconstruido, y se queda anclado a ese punto de la sala.
                # Si siguiera al avatar, este quedaria siempre al centro de la
                # pantalla y moverse en profundidad no se veria: justo lo que
                # hay que demostrar.
                if pivote is None:
                    pivote = np.mean(list(final_pts.values()), axis=0)

                glLoadIdentity()
                glTranslatef(0.0, 0.0, -distancia)
                glRotatef(elevacion, 1, 0, 0)
                glRotatef(orbita, 0, 1, 0)
                glTranslatef(-pivote[0], -pivote[1], -pivote[2])

                # Dibujar esqueleto 3D en OpenGL
                for a, b in CONEXIONES:
                    # Solo dibujar si ambos puntos están disponibles
                    if a in final_pts and b in final_pts: 
                        dibujar_hueso(final_pts[a], final_pts[b], color_final)
                        dibujar_joint(final_pts[a], color_final)
                        dibujar_joint(final_pts[b], color_final)
                
                # Dibujar cabeza si el punto 0 (nariz) está disponible
                if 0 in final_pts: 
                    dibujar_cabeza(final_pts[0], color_final)

# Actualizar pantalla
        pygame.display.flip()
        #Esta es la ventana de opengl donde se ve el avatar
# Mostrar AMBAS camaras con los puntos detectados encima.
# Sin esto no hay forma de saber cual de las dos no esta detectando: el avatar
# 3D solo se dibuja si ambas ven a la persona, asi que una sola camara fallando
# deja la ventana de OpenGL en negro sin explicar por que.
        vistas = []
        for etiqueta, frame, res in (("cam0", frame0, res0), ("cam1", frame1, res1)):
            v = cv2.resize(frame, (400, 300))
            if res.pose_landmarks:
                mp_draw.draw_landmarks(v, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            detecto = res.pose_landmarks is not None
            cv2.rectangle(v, (0, 0), (400, 24), (0, 0, 0), -1)
            cv2.putText(v, f"{etiqueta}: {'DETECTA' if detecto else 'sin persona'}",
                        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if detecto else (0, 0, 255), 1)
            vistas.append(v)

        panel = np.hstack(vistas)
        # Lectura metrica: la distancia absoluta a las camaras es el dato que
        # solo puede salir de la triangulacion. Sirve para demostrar en vivo
        # que el sistema mide y no estima.
        barra = np.zeros((58, panel.shape[1], 3), np.uint8)
        if dist_camara is not None:
            cv2.putText(barra, f"distancia a las camaras: {dist_camara:.0f} mm "
                               f"({dist_camara/1000:.2f} m)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(barra, "sin reconstruccion 3D (ambas camaras deben detectar)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        cv2.putText(barra, "flechas giran la vista 3D  |  Z/X acercan  |  R reinicia",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
        cv2.imshow("Camaras", np.vstack([barra, panel]))
        if cv2.waitKey(1) & 0xFF == 27: break

        # Salir con la tecla ESC
        
    cap0.release(); cap1.release(); pygame.quit()

if __name__ == "__main__":
    main()