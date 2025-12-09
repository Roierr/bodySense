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
from Herramientas import buscar_archivo_desesperadamente
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
    cfg = buscar_archivo_desesperadamente("avatar_config.json")
    if cfg:
        try:
            with open(cfg) as f:
                d = json.load(f)
                h = d["color_piel"].lstrip('#')
                color_final = tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))
        except: pass

    pygame.init()
    pygame.display.set_mode((1000, 800), DOUBLEBUF | OPENGL)
    gluPerspective(45, (1000/800), 0.1, 50000.0)
    
    # El zoom ajusta la distancia de la camara en el eje Z (-900.0) y la posicion vertical en el eje Y (-150.0) para que el avatar 3D, ya escalado, se vea centrado y completo dentro de la ventana de OpenGL.
    glTranslatef(0.0, -150.0, -900.0)
    glRotatef(180, 0, 1, 0)

    glDisable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST) 

    #  Establece la confianza mínima (0.1) que un punto corporal detectado por MediaPipe debe tener en ambas cámaras para ser considerado válido y usado en la reconstrucción 3D.
    TOLERANCIA_VISIBILIDAD = 0.1 

    mp_pose = mp.solutions.pose
    pose0 = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4, model_complexity=0)
    pose1 = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4, model_complexity=0)
    
    cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW); cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    #Esto es para ver el avatar mas grande en la pantalla
    ESCALA_GIGANTE = 1.5

    CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

    print("✅ AVATAR GIGANTE ACTIVADO.")

    while True:
        for event in pygame.event.get():
            if event.type == QUIT: pygame.quit(); return

        ret0, frame0 = cap0.read(); ret1, frame1 = cap1.read()
        if not ret0 or not ret1: continue

        res0 = pose0.process(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
        res1 = pose1.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if res0.pose_landmarks and res1.pose_landmarks:
            lm0, lm1 = res0.pose_landmarks.landmark, res1.pose_landmarks.landmark
            ptsL, ptsR, v_ids = [], [], []
            h, w, _ = frame0.shape
            
            for i in range(33):
                if lm0[i].visibility > TOLERANCIA_VISIBILIDAD and lm1[i].visibility > TOLERANCIA_VISIBILIDAD:
                    ptsL.append([lm0[i].x * w, lm0[i].y * h])
                    ptsR.append([lm1[i].x * w, lm1[i].y * h])
                    v_ids.append(i)
            
            if ptsL:
                p3d = cv2.triangulatePoints(PL, PR, np.array(ptsL).T, np.array(ptsR).T)
                p3d = (p3d[:3] / p3d[3]).T
                
                p3d = p3d * ESCALA_GIGANTE

                final_pts = {vid: [p[0], -p[1], p[2]] for vid, p in zip(v_ids, p3d)}
                
                for a, b in CONEXIONES:
                    if a in final_pts and b in final_pts: 
                        dibujar_hueso(final_pts[a], final_pts[b], color_final)
                        dibujar_joint(final_pts[a], color_final)
                        dibujar_joint(final_pts[b], color_final)
                
                if 0 in final_pts: 
                    dibujar_cabeza(final_pts[0], color_final)

        pygame.display.flip()
        cv2.imshow("Camara", cv2.resize(frame0, (300, 200)))
        if cv2.waitKey(1) & 0xFF == 27: break
        
    cap0.release(); cap1.release(); pygame.quit()

if __name__ == "__main__":
    main()